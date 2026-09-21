// ============================================================
//  LOGOS — GradientClip.cpp
//  Gradient Clipping — Exploding gradients rokta hai
//  Max norm clipping: if ||grad|| > threshold, scale karo
//  Threshold: 1.0 (standard for transformers)
// ============================================================
#ifndef LOGOS_GRADIENT_CLIP_CPP
#define LOGOS_GRADIENT_CLIP_CPP

#include "../include/Tensor.hpp"
#include <cmath>
#include <vector>
#include <iostream>

void clip_gradients(std::vector<Tensor*>& grads, float max_norm = 1.0f) {
    // Global L2 norm of all gradients
    float total_norm = 0.0f;
    for (const auto* g : grads)
        for (float v : g->data)
            total_norm += v * v;
    total_norm = std::sqrt(total_norm);

    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (auto* g : grads)
            for (float& v : g->data)
                v *= scale;
        // Uncomment for debug: 
        // std::cout << "✂️  Gradient clipped: norm=" << total_norm << " → " << max_norm << "\n";
    }

}

#endif
