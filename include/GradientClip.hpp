#pragma once
// ============================================================
//  LOGOS — GradientClip.hpp
//  Shock Wave Gradient Clipping (Rankine-Hugoniot)
//
//  Physics: gradient explosion = supersonic shock (Mach > 1)
//    M = ||g|| / max_norm
//    M < 1 : laminar flow   → no clipping
//    M >= 1: shock front    → rescale to sonic point (M=1)
//  Rankine-Hugoniot: scale = 1/M at shock boundary
// ============================================================
#include "Tensor.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

// ── Shock Wave Gradient Clipping ─────────────────────────────
// Physics analogy: gradient explosion = supersonic shock (Mach > 1)
// When ||g|| > max_norm → "Mach number" M = ||g||/max_norm > 1
// Rankine-Hugoniot condition: scale = 1/M (subsonic normalisation)
// Smooth pre-shock zone (M < 1): no clipping (laminar flow)
// Sharp shock front (M >= 1): hard rescale to sonic point
//
// Extra: per-layer Mach logging helps diagnose which layer shocks first
inline void clip_gradients(std::vector<Tensor*>& grads,
                           float max_norm = 1.0f,
                           bool  log_shock = false)
{
    // Compute global gradient norm (||g||)
    float total_norm = 0.0f;
    for (const auto* g : grads)
        for (float v : g->data) total_norm += v * v;
    total_norm = std::sqrt(total_norm);

    // Mach number: M = ||g|| / max_norm
    float mach = total_norm / (max_norm + 1e-6f);

    if (mach > 1.0f) {
        // Shock regime: Rankine-Hugoniot rescale → bring back to sonic point
        float scale = max_norm / (total_norm + 1e-6f);
        for (auto* g : grads)
            for (float& v : g->data) v *= scale;

        if (log_shock)
            std::cout << "[ShockClip] Mach=" << std::fixed << std::setprecision(2)
                      << mach << " > 1.0 → rescaled (scale=" << scale << ")\n";
    }
    // M < 1: laminar flow — no clipping needed
}

// Per-layer Mach number diagnostic (call separately for debugging)
inline float gradient_mach(const std::vector<Tensor*>& grads, float max_norm = 1.0f) {
    float norm = 0.0f;
    for (const auto* g : grads)
        for (float v : g->data) norm += v * v;
    return std::sqrt(norm) / (max_norm + 1e-6f);
}
