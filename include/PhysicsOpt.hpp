#pragma once
// ============================================================
//  LOGOS — PhysicsOpt.hpp
//  Langevin Dynamics Optimizer (Physics-based weight update)
//
//  Equation:
//    dW = -γ·∇L·dt + √(2γkT)·η
//  Where:
//    γ  = friction (0.9)    — controls momentum decay
//    ∇L = gradient of loss
//    T  = temperature       — annealed: T_start → T_end (cosine)
//    η  = Gaussian noise    — Brownian motion (thermal exploration)
//    dt = learning rate
//
//  Key design:
//    T_start = 0.1   (NOT 10.0 — high T causes noise explosion)
//    T_end   = 1e-5  (near-zero noise at convergence)
//    lr      = 1e-4  (conservative for stability)
//
//  ✅ This is the ONLY optimizer used in LOGOS.
//  ❌ Do NOT replace with Adam or any gradient-descent variant.
// ============================================================
#include "Tensor.hpp"
#include <cmath>
#include <vector>
#include <random>
#include <iostream>

class LangevinOptimizer {
public:
    float learning_rate;
    float friction;         // γ = 0.9
    float temperature;      // current T (annealed)
    float temp_start;       // T_start — keep LOW (0.1) to prevent explosion
    float temp_end;         // T_end   — near zero (1e-5)
    int   total_steps;
    int   current_step = 0;

    // Per-parameter velocity buffers (momentum)
    std::vector<std::vector<float>> velocity;

    std::mt19937 rng;
    std::normal_distribution<float> noise_dist{0.0f, 1.0f};

    // ── Constructor ──────────────────────────────────────────
    // lr=1e-4, T_start=0.1, T_end=1e-5 are stable defaults
    LangevinOptimizer(float lr       = 1e-4f,
                      float fric     = 0.9f,
                      float T_start  = 0.1f,    // LOW — prevents noise explosion
                      float T_end    = 1e-5f,
                      int   steps    = 100000,
                      int   seed     = 42)
        : learning_rate(lr), friction(fric),
          temperature(T_start), temp_start(T_start),
          temp_end(T_end), total_steps(steps),
          rng(seed)
    {}

    // Initialize velocity buffers for all params
    void init(const std::vector<Tensor*>& params) {
        velocity.clear();
        velocity.reserve(params.size());
        for (auto* p : params)
            velocity.emplace_back(p->total_size, 0.0f);
    }

    // Cosine annealing: T_start → T_end over total_steps
    void anneal() {
        float ratio  = std::min(1.0f, (float)current_step / (float)total_steps);
        float cos_v  = 0.5f * (1.0f + std::cos(3.14159265f * ratio));
        temperature  = temp_end + (temp_start - temp_end) * cos_v;
    }

    // One optimization step
    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads)
    {
        if (velocity.empty()) init(params);
        if (params.size() != grads.size())
            throw std::invalid_argument("params/grads size mismatch");

        anneal();

        // Noise scale = √(2γkT·dt)
        // Scaled by 0.01 extra to keep thermal kicks small
        float noise_scale = std::sqrt(2.0f * friction * temperature * learning_rate) * 0.01f;

        for (int pi = 0; pi < (int)params.size(); ++pi) {
            Tensor* W        = params[pi];
            const Tensor* G  = grads[pi];
            auto& vel        = velocity[pi];

            for (int i = 0; i < W->total_size; ++i) {
                float grad = G->data[i];

                // NaN / Inf guard
                if (std::isnan(grad) || std::isinf(grad)) grad = 0.0f;

                // Langevin update:
                // v_{t+1} = (1-γ)·v_t  - lr·∇L  + √(2γkT)·η
                float thermal = noise_scale * noise_dist(rng);
                vel[i] = (1.0f - friction) * vel[i]
                         - learning_rate * grad
                         + thermal;

                W->data[i] += vel[i];

                // Sanity guard: clamp extreme weights
                if (std::isnan(W->data[i]) || std::isinf(W->data[i]))
                    W->data[i] = 0.0f;
            }
        }
        ++current_step;
    }

    void log_state() const {
        std::cout << "LangevinOpt | step=" << current_step
                  << " T=" << std::scientific << temperature
                  << " lr=" << learning_rate << "\n";
    }
};
