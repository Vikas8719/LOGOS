#pragma once
//  Hybrid Stochastic Hamiltonian Mechanics:
//    Same physics as GPU shm_hybrid_kernel (VedicGEMM.cu)
//    Used in main.cpp / Kaggle / unit tests
//
//  v10-FIX: Three critical bugs fixed:
//
//  FIX-1: Temperature floor — T_end minimum clamped to 1e-3
//    Pehle: T_end = 1e-5 → T → 0.0000 at late training
//    Ab:    T_end >= 1e-3 enforced in anneal()
//    Reason: F = CE - T*S mein T=0 → entropy regularization OFF
//            → model overconfident on wrong predictions → loss spikes
//
//  FIX-2: AnurupyenaScaler target_rms lowered, max_scale tightened
//    Pehle: target_rms=1.0, max_scale=100 → grad_rms=0.003 → 350x boost → explosion
//    Ab:    target_rms=0.01, max_scale=10  → at most 10x rescale → stable
//
//  FIX-3: WeightPathIntegral best tracking fixed
//    Pehle: best_log_amplitude=0.0f → log_A always < 0 → best_step always 0
//    Ab:    best_log_amplitude = -1e30f (negative infinity) → always updated
//           Also added: reset_best() for re-initialization after warmup
// ============================================================
#include "Tensor.hpp"
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdint>

// ── FIX-1 constant: minimum temperature floor ─────────────────
// T below this causes F=CE-T*S to lose entropy regularization.
// 1e-3 keeps soft entropy signal alive throughout training.
static constexpr float T_MIN_FLOOR = 1e-3f;

// ── Free Energy Loss (CPU reference) ─────────────────────────
struct FreeEnergyLoss {
    static float compute(const float* logits, int vocab,
                         int target, float temperature,
                         float* grad_out)
    {
        // FIX-1: Clamp temperature — never let T drop to near-zero
        float T = std::max(temperature, T_MIN_FLOOR);

        float max_l = logits[0];
        for (int v=1; v<vocab; ++v) max_l = std::max(max_l, logits[v]);
        float sum = 0.0f;
        std::vector<float> p(vocab);
        for (int v=0; v<vocab; ++v) { p[v] = std::exp(logits[v]-max_l); sum+=p[v]; }
        for (int v=0; v<vocab; ++v) p[v] /= (sum + 1e-9f);

        float U = -std::log(std::max(p[target], 1e-9f));
        float S = 0.0f;
        for (int v=0; v<vocab; ++v)
            if (p[v] > 1e-12f) S -= p[v] * std::log(p[v]);

        float F = U - T * S;

        if (grad_out) {
            for (int v=0; v<vocab; ++v) {
                float y_v  = (v == target) ? 1.0f : 0.0f;
                float ce_g = p[v] - y_v;
                float H_g  = T * p[v] * (std::log(p[v]+1e-9f) + S);
                grad_out[v] = ce_g + H_g;
            }
        }
        return F;
    }
};

// ── Leapfrog Langevin Optimizer ───────────────────────────────
class LangevinOptimizer {
public:
    float learning_rate;
    float friction;
    float temperature;
    float temp_start;
    float temp_end;
    int   total_steps;
    int   current_step = 0;
    std::vector<std::vector<float>> velocity;
    std::mt19937 rng;
    std::normal_distribution<float> noise_dist{0.0f, 1.0f};

    LangevinOptimizer(float lr=1e-4f, float fric=0.9f,
                      float T_start=1e-5f, float T_end=1e-8f,
                      int steps=100000, int seed=42)
        : learning_rate(lr), friction(fric),
          temperature(T_start), temp_start(T_start),
          // FIX-1: enforce T_end floor
          temp_end(std::max(T_end, T_MIN_FLOOR)),
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
        // FIX-1: floor applied here too — can never reach 0
        temperature = std::max(T_MIN_FLOOR,
                               temp_end + (temp_start - temp_end) * cos_v);
    }

    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads)
    {
        if (velocity.empty()) init(params);
        anneal();
        float noise_scale = std::sqrt(friction * temperature * learning_rate * 0.5f);
        for (int pi=0; pi<(int)params.size(); ++pi) {
            Tensor* W       = params[pi];
            const Tensor* G = grads[pi];
            auto& vel       = velocity[pi];
            for (int i=0; i<W->total_size; ++i) {
                float grad = G->data[i];
                if (std::isnan(grad)||std::isinf(grad)) grad=0.0f;
                float thermal = noise_scale * noise_dist(rng);
                float v_half  = friction * vel[i]
                              - (learning_rate * 0.5f) * grad + thermal;
                float w_new   = W->data[i] + learning_rate * v_half;
                if (std::isnan(w_new)||std::isinf(w_new)) w_new = W->data[i];
                vel[i]     = v_half;
                W->data[i] = w_new;
            }
        }
        ++current_step;
    }

    void log_state() const {
        std::cout << "LangevinOpt [Leapfrog v7]"
                  << " | step=" << current_step
                  << " | T=" << std::scientific << temperature
                  << " | lr=" << learning_rate << "\n";
    }
};

// ── Hybrid SHM Optimizer ──────────────────────────────────────
class HybridSHMOptimizer {
public:
    float learning_rate;
    float friction;
    float mom_decay;
    float temperature, temp_start, temp_end;
    float alpha_H_start, alpha_H_end;
    int   total_steps;
    int   current_step = 0;
    float alpha_H = 0.3f;
    float alpha_L = 0.7f;

    std::vector<std::vector<float>> velocity;
    std::mt19937 rng;
    std::normal_distribution<float> noise_dist{0.0f, 1.0f};

    // v10-HAM: Hamiltonian-dominant defaults
    // T_start=0.5 (real annealing), aH_start=0.7 (gradient-strong from start)
    HybridSHMOptimizer(float lr=1e-4f, float fric=0.1f, float mom_d=0.95f,
                       float T_start=0.5f, float T_end=1e-3f,
                       float aH_start=0.7f, float aH_end=0.99f,
                       int steps=100000, int seed=42)
        : learning_rate(lr), friction(fric), mom_decay(mom_d),
          temperature(T_start), temp_start(T_start),
          // FIX-1: enforce T_end floor — never zero
          temp_end(std::max(T_end, T_MIN_FLOOR)),
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
        // FIX-1: temperature floor — entropy reg always active
        temperature = std::max(T_MIN_FLOOR,
                               temp_end + (temp_start - temp_end) * cos_v);
        alpha_H = alpha_H_start + (alpha_H_end - alpha_H_start) * (1.0f - cos_v);
        alpha_L = 1.0f - alpha_H;
    }

    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads)
    {
        if (velocity.empty()) init(params);
        anneal();
        float noise_scale = std::sqrt(friction * temperature * learning_rate * alpha_L);
        for (int pi=0; pi<(int)params.size(); ++pi) {
            Tensor* W       = params[pi];
            const Tensor* G = grads[pi];
            auto& vel       = velocity[pi];
            for (int i=0; i<W->total_size; ++i) {
                float grad = G->data[i];
                if (std::isnan(grad)||std::isinf(grad)) grad=0.0f;
                float thermal       = noise_scale * noise_dist(rng);
                float ham_kick      = -(alpha_H * learning_rate * 0.5f) * grad;
                float lang_friction = -(alpha_L * learning_rate * 0.5f) * friction * vel[i];
                float v_half        = mom_decay * vel[i] + ham_kick + lang_friction + thermal;
                float w_new         = W->data[i] + learning_rate * v_half;
                if (std::isnan(w_new)||std::isinf(w_new)) w_new = W->data[i];
                vel[i]     = v_half;
                W->data[i] = w_new;
            }
        }
        ++current_step;
    }

    void log_state() const {
        std::cout << "HybridSHMOpt [v9-FIX]"
                  << " | step=" << current_step
                  << " | T="    << std::scientific << temperature
                  << " | α_H="  << std::fixed << std::setprecision(2) << alpha_H
                  << " | α_L="  << alpha_L
                  << " | lr="   << learning_rate << "\n";
    }
};

// ── Natural Gradient Optimizer ────────────────────────────────
class NaturalGradientOptimizer {
public:
    float learning_rate;
    float beta;
    float epsilon;
    float mom_decay;
    float temperature;
    float temp_start, temp_end;
    int   total_steps;
    int   current_step = 0;
    std::vector<std::vector<float>> velocity;
    std::vector<std::vector<float>> fisher_diag;
    std::mt19937 rng;
    std::normal_distribution<float> noise_dist{0.0f, 1.0f};

    NaturalGradientOptimizer(float lr=1e-4f, float ema_beta=0.99f,
                              float eps=1e-8f, float mom=0.9f,
                              float T_start=0.01f, float T_end=1e-3f,
                              int steps=100000, int seed=42)
        : learning_rate(lr), beta(ema_beta), epsilon(eps), mom_decay(mom),
          temperature(T_start), temp_start(T_start),
          temp_end(std::max(T_end, T_MIN_FLOOR)),   // FIX-1
          total_steps(steps), rng(seed)
    {}

    void init(const std::vector<Tensor*>& params) {
        velocity.clear(); fisher_diag.clear();
        for (auto* p : params) {
            velocity.emplace_back(p->total_size, 0.0f);
            fisher_diag.emplace_back(p->total_size, epsilon);
        }
    }

    void anneal() {
        float ratio = std::min(1.0f, (float)current_step / (float)total_steps);
        float cos_v = 0.5f * (1.0f + std::cos(3.14159265f * ratio));
        temperature = std::max(T_MIN_FLOOR,
                               temp_end + (temp_start - temp_end) * cos_v);
    }

    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads)
    {
        if (velocity.empty()) init(params);
        anneal();
        float noise_scale = std::sqrt(2.0f * temperature * learning_rate);
        for (int pi=0; pi<(int)params.size(); ++pi) {
            Tensor* W = params[pi]; const Tensor* G = grads[pi];
            auto& vel = velocity[pi]; auto& F = fisher_diag[pi];
            for (int i=0; i<W->total_size; ++i) {
                float g = G->data[i];
                if (std::isnan(g)||std::isinf(g)) g=0.0f;
                F[i] = beta*F[i] + (1.0f-beta)*g*g;
                float g_natural = g / (std::sqrt(F[i]) + epsilon);
                float v_half    = mom_decay*vel[i]
                                - (learning_rate*0.5f)*g_natural
                                + noise_scale*noise_dist(rng);
                float w_new     = W->data[i] + learning_rate*v_half;
                if (std::isnan(w_new)||std::isinf(w_new)) w_new = W->data[i];
                vel[i]=v_half; W->data[i]=w_new;
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
struct WeightPathIntegral {
    float hbar;
    int   n_history;

    float log_amplitude      = 0.0f;
    float cumulative_action  = 0.0f;
    int   step_count         = 0;

    struct PathStep {
        float loss, step_norm, action, log_amplitude;
    };
    std::vector<PathStep> history;

    // FIX-3: Initialize to -infinity so ANY first step becomes best_step
    float best_log_amplitude = -1e30f;
    int   best_step          = 0;

    explicit WeightPathIntegral(float hbar_=1.0f, int hist=1000)
        : hbar(std::max(1e-6f, hbar_)), n_history(hist)
    { history.reserve(hist); }

    // FIX-3: reset_best() — call after warmup to restart best tracking
    // This prevents the very first bad steps from dominating best_step forever.
    // GPU path: record ||delta_theta|| directly and track an EMA of action (self-normalising LR signal).
    float ema_action  = 0.0f;
    float last_action = 0.0f;
    bool  ema_ready   = false;
    void record_step_norm(float loss, float step_norm) {
        float action = std::max(loss, 0.0f) * step_norm;
        cumulative_action += action;
        log_amplitude     -= action / hbar;
        ema_action  = ema_ready ? 0.98f * ema_action + 0.02f * action : action;
        ema_ready   = true;
        last_action = action;
        ++step_count;
    }
    // LR multiplier in [lr_min_frac,1]: shrinks when current action spikes above its running average.
    float lr_scale_ema(float lr_min_frac = 0.5f) const {
        float rel = std::min(1.0f, std::exp(-(last_action - ema_action) / (ema_action + 1e-8f)));
        return lr_min_frac + (1.0f - lr_min_frac) * rel;
    }

    void reset_best() {
        best_log_amplitude = log_amplitude;
        best_step          = step_count;
    }

    void record_step(float loss, const std::vector<float>& delta_theta) {
        float step_norm = 0.0f;
        for (float d : delta_theta) step_norm += d*d;
        step_norm = std::sqrt(step_norm);

        float action = loss * step_norm;
        log_amplitude     -= action / hbar;
        cumulative_action += action;

        // FIX-3: now correctly triggers because best starts at -1e30
        if (log_amplitude > best_log_amplitude) {
            best_log_amplitude = log_amplitude;
            best_step          = step_count;
        }

        PathStep ps{loss, step_norm, action, log_amplitude};
        if ((int)history.size() < n_history) {
            history.push_back(ps);
        } else {
            history.erase(history.begin());
            history.push_back(ps);
        }
        ++step_count;
    }

    void record_step_from_tensors(float loss,
                                   const std::vector<Tensor*>& params_after,
                                   const std::vector<float>&   snapshot_before) {
        std::vector<float> delta;
        delta.reserve(snapshot_before.size());
        int offset = 0;
        for (const auto* p : params_after) {
            for (int i=0; i<p->total_size; ++i)
                delta.push_back(p->data[i] - snapshot_before[offset+i]);
            offset += p->total_size;
        }
        record_step(loss, delta);
    }

    static std::vector<float> snapshot(const std::vector<Tensor*>& params) {
        int total=0;
        for (const auto* p : params) total+=p->total_size;
        std::vector<float> snap; snap.reserve(total);
        for (const auto* p : params)
            snap.insert(snap.end(), p->data.begin(), p->data.end());
        return snap;
    }

    float amplitude()          const { return std::exp(std::max(log_amplitude, -88.0f)); }
    float relative_amplitude() const {
        if (best_log_amplitude <= -1e29f) return 1.0f;
        float delta = log_amplitude - best_log_amplitude;
        return std::exp(std::max(delta, -88.0f));
    }
    float lr_scale(float lr_min_frac=0.1f) const {
        float rel = relative_amplitude();
        return lr_min_frac + (1.0f - lr_min_frac) * rel;
    }
    float recent_mean_action(int last_n=50) const {
        if (history.empty()) return 0.0f;
        int start = std::max(0, (int)history.size()-last_n);
        float sum=0.0f;
        for (int i=start; i<(int)history.size(); ++i) sum+=history[i].action;
        return sum / (float)(history.size()-start);
    }

    void log_state() const {
        std::cout << "WeightPathIntegral"
                  << " | step=" << step_count
                  << " | log_A=" << std::fixed << std::setprecision(3) << log_amplitude
                  << " | rel_A=" << relative_amplitude()
                  << " | Σaction=" << cumulative_action
                  << " | best_step=" << best_step
                  << " | hbar=" << hbar << "\n";
    }
};

// NOTE: ReynoldsBatchNorm is defined in LayerNorm.hpp
// NOTE: FeynmanDropout    is defined in FeedForward.hpp
