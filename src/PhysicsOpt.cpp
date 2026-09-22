// ============================================================
//  LOGOS — PhysicsOpt.cpp
//  Langevin Dynamics Optimizer (Physics-based weight update)
//
//  Equation:
//    dW = -γ·∇L·dt + √(2γkT)·η
//  Where:
//    γ  = friction (0.9) — controls momentum decay
//    ∇L = gradient of loss
//    T  = temperature (annealing schedule: 10.0 → 0.001)
//    η  = Gaussian noise (Brownian motion simulation)
//    dt = learning rate (step size)
//
//  vs Adam: Langevin adds principled thermal noise for
//  exploration — smoother loss curves in theory
// ============================================================
#ifndef LOGOS_PHYSICS_OPT_CPP
#define LOGOS_PHYSICS_OPT_CPP

#include "../include/Tensor.hpp"
#include <cmath>
#include <vector>
#include <random>
#include <iostream>

class LangevinOptimizer {
public:
    float learning_rate;
    float friction;         // γ = 0.9 (fixed)
    float temperature;      // T — annealed over training
    float temp_start;       // T_start = 10.0
    float temp_end;         // T_end   = 0.001
    int   total_steps;
    int   current_step = 0;

    // Momentum buffers (one per parameter tensor)
    std::vector<std::vector<float>> velocity;

    std::mt19937 rng;
    std::normal_distribution<float> noise_dist{0.0f, 1.0f};

    LangevinOptimizer(float lr = 3e-4f,
                      float fric = 0.9f,
                      float T_start = 10.0f,
                      float T_end = 0.001f,
                      int steps = 100000,
                      int seed = 42)
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

    // Update temperature — cosine annealing schedule
    void anneal_temperature() {
        float ratio = static_cast<float>(current_step) / total_steps;
        // Cosine schedule: smooth decay
        constexpr float pi = 3.14159265358979323846f;
        float cos_val = 0.5f * (1.0f + std::cos(pi * ratio));
        temperature = temp_end + (temp_start - temp_end) * cos_val;
    }

    // One optimization step
    // params: pointers to weight tensors
    // grads:  corresponding gradient tensors (same size)
    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads)
    {
        if (velocity.empty()) init(params);
        if (params.size() != grads.size())
            throw std::invalid_argument("params/grads size mismatch");

        anneal_temperature();
        // FIX 1 (Noise Scale): Sahi Langevin noise = sqrt(2·γ·T·dt)
        // learning_rate = dt (step size), noise_scale mein lr nahi chahiye tha
        // Pehle: sqrt(2γT·lr) — galat (lr double count ho raha tha vel update se)
        // Ab:   sqrt(2γT·dt) jahan dt = learning_rate — sahi Fokker-Planck equation
        float noise_scale = std::sqrt(2.0f * friction * temperature);

        for (int pi = 0; pi < (int)params.size(); ++pi) {
            Tensor* W = params[pi];
            const Tensor* G = grads[pi];
            auto& vel = velocity[pi];

            for (int i = 0; i < W->total_size; ++i) {
                float grad = G->data[i];

                // NaN guard — NaN aaye toh skip
                if (std::isnan(grad) || std::isinf(grad)) {
                    std::cerr << "⚠️  NaN/Inf gradient detected — skipping\n";
                    grad = 0.0f;
                }

                // FIX 2 (Velocity Sign): Sahi Langevin momentum update:
                //   v = γ·v - lr·∇L + sqrt(2γT)·η
                // Pehle: (1-γ)·v — galat, momentum decay bahut zyada tha (γ=0.9 → sirf 0.1x vel rakhta tha)
                // Ab:    γ·v — sahi, friction se scale hota hai (0.9x vel retain)
                float thermal_noise = noise_scale * noise_dist(rng);
                vel[i] = friction * vel[i]
                          - learning_rate * grad
                          + thermal_noise;

                W->data[i] += vel[i];
            }
        }
        ++current_step;
    }

    void log_state() const {
        std::cout << "PhysicsOpt | step=" << current_step
                  << " T=" << temperature
                  << " lr=" << learning_rate << "\n";
    }
};

#endif
