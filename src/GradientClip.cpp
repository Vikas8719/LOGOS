
#ifndef LOGOS_GRADIENT_CLIP_CPP
#define LOGOS_GRADIENT_CLIP_CPP

#include "../include/Tensor.hpp"
#include <cmath>
#include <vector>
#include <iostream>

void clip_gradients(std::vector<Tensor*>& grads, float max_norm = 1.0f) {
    
    float total_norm = 0.0f;
    for (const auto* g : grads)
        for (float v : g->data)
            total_norm += v * v;
    total_norm = std::sqrt(total_norm);

    if (!std::isfinite(total_norm) || total_norm == 0.0f) return;

    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (auto* g : grads)
            for (float& v : g->data)
                v *= scale;
    }

}

#endif
