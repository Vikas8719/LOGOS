#pragma once
// ============================================================
//  LOGOS — PhysicsOpt.hpp  (Langevin Dynamics Optimizer)
// ============================================================
#include "Tensor.hpp"
#include <cmath>
#include <vector>
#include <random>
#include <iostream>

class LangevinOptimizer {
public:
    float learning_rate, friction, temperature, temp_start, temp_end;
    int total_steps, current_step = 0;
    std::vector<std::vector<float>> velocity;
    std::mt19937 rng;
    std::normal_distribution<float> noise_dist{0.0f, 1.0f};

    LangevinOptimizer(float lr=3e-4f, float fric=0.9f,
                      float T_start=10.0f, float T_end=0.001f,
                      int steps=100000, int seed=42)
        : learning_rate(lr), friction(fric),
          temperature(T_start), temp_start(T_start),
          temp_end(T_end), total_steps(steps), rng(seed)
    {}

    void init(const std::vector<Tensor*>& params) {
        velocity.clear();
        for (auto* p : params)
            velocity.emplace_back(p->total_size, 0.0f);
    }

    void anneal() {
        float ratio = (float)current_step / total_steps;
        float cos_v = 0.5f * (1.0f + std::cos(3.14159265f * ratio));
        temperature = temp_end + (temp_start - temp_end) * cos_v;
    }

    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads)
    {
        if (velocity.empty()) init(params);
        anneal();
        float noise_scale = std::sqrt(2.0f * friction * temperature * learning_rate) * 0.01f;  // scale down noise
        for (int pi = 0; pi < (int)params.size(); ++pi) {
            Tensor* W = params[pi];
            const Tensor* G = grads[pi];
            auto& vel = velocity[pi];
            for (int i = 0; i < W->total_size; ++i) {
                float grad = G->data[i];
                if (std::isnan(grad) || std::isinf(grad)) grad = 0.0f;
                float noise = noise_scale * noise_dist(rng);
                vel[i] = (1.0f - friction) * vel[i] - learning_rate * grad + noise;
                W->data[i] += vel[i];
            }
        }
        ++current_step;
    }
};
