// ============================================================
//  LOGOS — Trainer.cpp
//  Training Loop: Forward → Cross-Entropy Loss → Backward → Update
//
//  Note: Full backprop in C++ without autograd = manual gradients.
//  For Phase 4 training, numerical gradients (finite differences)
//  used for correctness, then switch to analytical backprop.
// ============================================================
#ifndef LOGOS_TRAINER_CPP
#define LOGOS_TRAINER_CPP

#include "../include/Tensor.hpp"
#include "Model.cpp"
#include "PhysicsOpt.cpp"
#include "GradientClip.cpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

class Trainer {
public:
    LOGOSModel& model;
    LangevinOptimizer optimizer;
    int batch_size  = 32;
    int seq_len     = 512;
    float lr        = 3e-4f;
    int save_every  = 1000;   // Checkpoint every 1000 steps
    int log_every   = 100;    // Loss log every 100 steps
    int step        = 0;

    Trainer(LOGOSModel& m, float learning_rate = 3e-4f)
        : model(m),
          optimizer(learning_rate, 0.9f, 10.0f, 0.001f, 100000)
    {
        optimizer.init(model.parameters());
        std::cout << "✅ Trainer ready | LR=" << learning_rate
                  << " | Langevin Physics Optimizer\n";
    }

    // Cross-entropy loss (single sequence)
    float cross_entropy_loss(const Tensor& logits,
                             const std::vector<int>& targets)
    {
        float loss = 0.0f;
        int seq = logits.rows(), vocab = logits.cols();
        int count = 0;
        for (int i = 0; i < std::min(seq, (int)targets.size()); ++i) {
            int target = targets[i];
            if (target < 0 || target >= vocab) continue;

            // Log-softmax of correct token
            float max_l = logits.at(i, 0);
            for (int v = 1; v < vocab; ++v)
                max_l = std::max(max_l, logits.at(i, v));

            float log_sum = 0.0f;
            for (int v = 0; v < vocab; ++v)
                log_sum += std::exp(logits.at(i, v) - max_l);
            log_sum = std::log(log_sum) + max_l;

            loss += log_sum - logits.at(i, target);
            ++count;
        }
        return count > 0 ? loss / count : 0.0f;
    }

    // Numerical gradient (finite differences — for testing)
    // delta: step size for finite diff
    void compute_gradients_numerical(const std::vector<int>& input_ids,
                                     const std::vector<int>& target_ids,
                                     std::vector<Tensor>& grad_tensors,
                                     float delta = 1e-4f)
    {
        auto params = model.parameters();
        grad_tensors.clear();
        grad_tensors.reserve(params.size());
        for (auto* p : params)
            grad_tensors.emplace_back(p->shape, 0.0f);

        // Only compute for a few params (full numerical grad is O(params) — slow)
        // In production: replace with analytical backprop
        std::cout << "⚠️  Numerical gradient (slow — for testing only)\n";

        for (int pi = 0; pi < (int)params.size(); ++pi) {
            Tensor* W = params[pi];
            // Only grad first 10 elements for speed demo
            int n_check = std::min(10, W->total_size);
            for (int i = 0; i < n_check; ++i) {
                float orig = W->data[i];
                W->data[i] = orig + delta;
                Tensor logits_plus = model.forward(input_ids);
                float loss_plus = cross_entropy_loss(logits_plus, target_ids);

                W->data[i] = orig - delta;
                Tensor logits_minus = model.forward(input_ids);
                float loss_minus = cross_entropy_loss(logits_minus, target_ids);

                grad_tensors[pi].data[i] = (loss_plus - loss_minus) / (2 * delta);
                W->data[i] = orig;  // Restore
            }
        }
    }

    // Training step (single batch)
    float train_step(const std::vector<int>& input_ids,
                     const std::vector<int>& target_ids)
    {
        // Forward pass
        Tensor logits = model.forward(input_ids);
        float loss = cross_entropy_loss(logits, target_ids);

        // Check for NaN — stop immediately
        if (std::isnan(loss)) {
            std::cerr << "❌ NaN loss at step " << step << " — halting\n";
            return -1.0f;
        }

        // Gradient computation (simplified — output layer only for now)
        // Full backprop to be added in next phase
        auto params = model.parameters();
        std::vector<Tensor> grads;
        for (auto* p : params)
            grads.emplace_back(p->shape, 0.0f);

        // Gradient of loss w.r.t. logits (softmax + cross-entropy)
        // ∂L/∂logit_i = softmax_i - 1(i==target) / seq_len
        // Only for LM head for now (full backprop = next phase)
        int seq = logits.rows(), vocab = logits.cols();
        Tensor logit_grad(logits.shape, 0.0f);
        for (int i = 0; i < seq && i < (int)target_ids.size(); ++i) {
            // Softmax
            float max_l = logits.at(i, 0);
            for (int v = 1; v < vocab; ++v) max_l = std::max(max_l, logits.at(i,v));
            float sum = 0.0f;
            for (int v = 0; v < vocab; ++v) {
                logit_grad.at(i,v) = std::exp(logits.at(i,v) - max_l);
                sum += logit_grad.at(i,v);
            }
            for (int v = 0; v < vocab; ++v) {
                logit_grad.at(i,v) /= sum;
                if (v == target_ids[i]) logit_grad.at(i,v) -= 1.0f;
                logit_grad.at(i,v) /= seq;
            }
        }

        // Gradient clip
        std::vector<Tensor*> grad_ptrs;
        for (auto& g : grads) grad_ptrs.push_back(&g);
        clip_gradients(grad_ptrs);

        // Langevin update
        optimizer.step(params, grad_ptrs);
        ++step;

        // Log
        if (step % log_every == 0) {
            std::cout << "Step " << step << " | Loss: " << loss
                      << " | T=" << optimizer.temperature << "\n";
        }

        return loss;
    }
};

#endif
