#pragma once
#include "Tensor.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

inline void clip_gradients(std::vector<Tensor*>& grads,
                           float max_norm = 1.0f,
                           bool  log_shock = false)
{
    float total_norm = 0.0f;
    for (const auto* g : grads) {
        for (float v : g->data) {
            if (std::isnan(v) || std::isinf(v)) { total_norm = 0.0f; goto zero_grads; }
            total_norm += v * v;
        }
    }
    total_norm = std::sqrt(total_norm);
    {
        float mach = total_norm / (max_norm + 1e-6f);
        if (mach > 1.0f) {
            float scale = max_norm / (total_norm + 1e-6f);
            for (auto* g : grads)
                for (float& v : g->data) v *= scale;
            if (log_shock)
                std::cout << "[ShockClip] Mach=" << std::fixed << std::setprecision(2)
                          << mach << " > 1.0 → rescaled (scale=" << scale << ")\n";
        }
    }
    return;
zero_grads:
    for (auto* g : grads)
        for (float& v : g->data) v = 0.0f;
    if (log_shock)
        std::cout << "[ShockClip] NaN/Inf gradient detected — zeroed all grads\n";
}

inline float gradient_mach(const std::vector<Tensor*>& grads, float max_norm = 1.0f) {
    float norm = 0.0f;
    for (const auto* g : grads)
        for (float v : g->data) {
            if (std::isnan(v) || std::isinf(v)) return 0.0f;
            norm += v * v;
        }
    return std::sqrt(norm) / (max_norm + 1e-6f);
}
