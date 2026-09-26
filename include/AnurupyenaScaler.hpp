#pragma once
// ============================================================
//  LOGOS — AnurupyenaScaler.hpp
//  Vedic Sutra: Āṇurūpyeṇa (अणुरूप्येण)
//  "Proportionality / In Proportion To"
//
//  Meaning:
//    "By suitable / by the proportion" — when a scaling factor is
//    needed, choose one proportional to the magnitude of the quantity
//    being scaled, rather than a fixed constant.
//
//  Application to Gradient Scaling:
//    Standard gradient descent: θ -= lr * g
//    Problem: different parameter groups (embedding, attention, FFN,
//    layer-norm) have wildly different gradient magnitudes. A single
//    lr clips or under-updates different groups disproportionately.
//
//    Āṇurūpyeṇa fix:
//      scale_i = (target_rms) / (rms(g_i) + ε)
//      g̃_i    = g_i * scale_i   [proportional rescaling]
//    where target_rms is a global reference RMS (default = 1.0).
//
//    This is different from:
//      • Adam/RMSProp:  per-element second moment → over-adaptive
//      • Gradient clip: hard cap, not proportional rescaling
//      • NaturalGrad:   Fisher metric (more expensive)
//
//    Āṇurūpyeṇa is PER-TENSOR proportional:
//      each weight tensor's gradient is scaled so its RMS matches
//      target_rms. Tensors that are already at target_rms are
//      unchanged. Over-large gradients are shrunk, tiny ones are
//      boosted — all proportionally.
//
//  Implementation note:
//    Applied AFTER clip_gradients() and BEFORE the optimizer step.
//    Works with any optimizer (SHM, Langevin, NaturalGrad).
//    Maintains the gradient DIRECTION; only magnitude changes.
//    Safe to combine with RiemannianMetric (apply Anurupyena first).
//
//  Reference: Vedic Mathematics (Tirthaji, 1965), Sutra 13
//             "Sopaantyadvayamantyam" family — proportional methods
// ============================================================
#include "Tensor.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

class AnurupyenaScaler {
public:
    float target_rms;   // reference RMS all tensors are scaled to
    float epsilon;      // numerical safety (avoid div-by-zero)
    float min_scale;    // minimum scale factor (prevent explosion on tiny grads)
    float max_scale;    // maximum scale factor (prevent explosion on large grads)
    bool  enabled;      // can be toggled per-run

    // EMA tracking of actual grad RMS per tensor (for diagnostics)
    float ema_beta;
    std::vector<float> rms_ema;   // one per param tensor
    bool  ema_init = false;

    AnurupyenaScaler(float target    = 1.0f,
                     float eps       = 1e-8f,
                     float min_s     = 0.01f,
                     float max_s     = 100.0f,
                     float ema_b     = 0.99f,
                     bool  on        = true)
        : target_rms(target), epsilon(eps),
          min_scale(min_s), max_scale(max_s),
          enabled(on), ema_beta(ema_b)
    {}

    // ── scale_gradients ──────────────────────────────────────
    // Apply Āṇurūpyeṇa proportional rescaling to a list of gradient tensors.
    // Modifies grads in-place.
    //
    // For each gradient tensor g_i:
    //   rms_i    = sqrt( mean(g_i^2) )         [per-tensor RMS]
    //   scale_i  = target_rms / (rms_i + eps)  [proportional factor]
    //   scale_i  = clamp(scale_i, min_s, max_s) [safety]
    //   g̃_i = g_i * scale_i                    [in-place rescale]
    void scale_gradients(std::vector<Tensor*>& grads) {
        if (!enabled) return;

        if (!ema_init) {
            rms_ema.assign(grads.size(), target_rms);
            ema_init = true;
        }
        if (rms_ema.size() != grads.size())
            rms_ema.resize(grads.size(), target_rms);

        for (int gi = 0; gi < (int)grads.size(); ++gi) {
            Tensor* g = grads[gi];
            if (!g || g->total_size == 0) continue;

            // ── Compute per-tensor RMS ────────────────────────
            float sum_sq = 0.0f;
            for (int i = 0; i < g->total_size; ++i)
                sum_sq += g->data[i] * g->data[i];
            float rms = std::sqrt(sum_sq / (float)(g->total_size) + epsilon);

            // EMA tracking
            rms_ema[gi] = ema_beta * rms_ema[gi] + (1.0f - ema_beta) * rms;

            // ── Proportional scale (Āṇurūpyeṇa) ─────────────
            // If rms < ε: tensor is essentially zero, skip to avoid
            // division producing infinity that pollutes the optimizer.
            if (rms < epsilon * 10.0f) continue;

            float scale = target_rms / rms;
            scale = std::max(min_scale, std::min(max_scale, scale));

            // ── Rescale in-place ──────────────────────────────
            for (int i = 0; i < g->total_size; ++i)
                g->data[i] *= scale;
        }
    }

    // ── Diagnostic log ───────────────────────────────────────
    void log_state(int step = -1) const {
        if (!ema_init) { std::cout << "AnurupyenaScaler: not yet applied\n"; return; }
        float min_r = 1e30f, max_r = 0.0f, mean_r = 0.0f;
        for (float r : rms_ema) {
            min_r  = std::min(min_r, r);
            max_r  = std::max(max_r, r);
            mean_r += r;
        }
        if (!rms_ema.empty()) mean_r /= (float)rms_ema.size();
        std::cout << "AnurupyenaScaler [Vedic proportional grad scale]"
                  << (step >= 0 ? " | step=" + std::to_string(step) : "")
                  << " | target_rms=" << std::fixed << std::setprecision(4) << target_rms
                  << " | grad_rms(min=" << min_r
                  << " mean=" << mean_r
                  << " max=" << max_r << ")\n";
    }
};
