#pragma once
// ============================================================
#include "Tensor.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

class AnurupyenaScaler {
public:
    float target_rms;   // FIX-2: was 1.0 → now 0.01
    float epsilon;
    float min_scale;
    float max_scale;    // FIX-2: was 100.0 → now 10.0
    bool  enabled;
    float ema_beta;
    std::vector<float> rms_ema;
    bool  ema_init = false;

    AnurupyenaScaler(float target  = 0.01f,   // FIX-2: realistic default
                     float eps     = 1e-8f,
                     float min_s   = 0.01f,
                     float max_s   = 10.0f,   // FIX-2: tighter clamp
                     float ema_b   = 0.99f,
                     bool  on      = true)
        : target_rms(target), epsilon(eps),
          min_scale(min_s), max_scale(max_s),
          enabled(on), ema_beta(ema_b)
    {}

    void scale_gradients(std::vector<Tensor*>& grads) {
        if (!enabled) return;

        if (!ema_init) {
            rms_ema.assign(grads.size(), target_rms);
            ema_init = true;
        }
        if (rms_ema.size() != grads.size())
            rms_ema.resize(grads.size(), target_rms);

        for (int gi=0; gi<(int)grads.size(); ++gi) {
            Tensor* g = grads[gi];
            if (!g || g->total_size==0) continue;

            float sum_sq = 0.0f;
            for (int i=0; i<g->total_size; ++i)
                sum_sq += g->data[i] * g->data[i];
            float rms = std::sqrt(sum_sq / (float)g->total_size + epsilon);

            rms_ema[gi] = ema_beta * rms_ema[gi] + (1.0f-ema_beta) * rms;

            // Skip near-zero tensors to avoid div-by-zero boost
            if (rms < epsilon * 10.0f) continue;

            // FIX-2: scale = target / rms, clamped to [min_s, max_s=10]
            // With target=0.01 and typical rms=0.003: scale=3.3x (safe)
            // With target=0.01 and large rms=0.1:    scale=0.1x (shrinks, safe)
            float scale = target_rms / rms;
            scale = std::max(min_scale, std::min(max_scale, scale));

            for (int i=0; i<g->total_size; ++i)
                g->data[i] *= scale;
        }
    }

    void log_state(int step=-1) const {
        if (!ema_init) { std::cout << "AnurupyenaScaler: not yet applied\n"; return; }
        float min_r=1e30f, max_r=0.0f, mean_r=0.0f;
        for (float r : rms_ema) {
            min_r  = std::min(min_r, r);
            max_r  = std::max(max_r, r);
            mean_r += r;
        }
        if (!rms_ema.empty()) mean_r /= (float)rms_ema.size();
        std::cout << "AnurupyenaScaler [FIX-2 | target=" << target_rms
                  << " max_scale=" << max_scale << "]"
                  << (step>=0 ? " | step="+std::to_string(step) : "")
                  << " | grad_rms(min=" << std::fixed << std::setprecision(5) << min_r
                  << " mean=" << mean_r
                  << " max=" << max_r << ")\n";
    }
};
