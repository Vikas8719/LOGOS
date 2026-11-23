#pragma once
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

// Tracks step "action" (loss * update norm) and derives an adaptive LR multiplier from it.
struct WeightPathIntegral {
    float hbar;
    float log_amplitude     = 0.0f;
    float cumulative_action = 0.0f;
    float ema_action        = 0.0f;
    float last_action       = 0.0f;
    float best_log_amplitude = -1e30f;
    int   step_count = 0;
    int   best_step  = 0;
    bool  ema_ready  = false;

    explicit WeightPathIntegral(float hbar_ = 1.0f) : hbar(std::max(1e-6f, hbar_)) {}

    void record_step_norm(float loss, float step_norm) {
        const float action = std::max(loss, 0.0f) * std::max(step_norm, 0.0f);
        // NaN/Inf loss ya norm se state hamesha ke liye poison na ho (std::max(NaN,0)=NaN)
        if (!std::isfinite(action)) return;
        cumulative_action += action;
        log_amplitude     -= action / hbar;
        ema_action  = ema_ready ? 0.98f * ema_action + 0.02f * action : action;
        ema_ready   = true;
        last_action = action;
        ++step_count;
    }

    float lr_scale_ema(float lr_min_frac = 0.5f) const {
        const float rel = std::min(1.0f, std::exp(-(last_action - ema_action) / (ema_action + 1e-8f)));
        return lr_min_frac + (1.0f - lr_min_frac) * rel;
    }

    void reset_best() {
        best_log_amplitude = log_amplitude;
        best_step          = step_count;
    }

    float amplitude() const { return std::exp(std::max(log_amplitude, -88.0f)); }

    float relative_amplitude() const {
        if (best_log_amplitude <= -1e29f) return 1.0f;
        return std::exp(std::max(log_amplitude - best_log_amplitude, -88.0f));
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
