#pragma once
// ============================================================
#include "Tensor.hpp"
#include "VedicGEMM.hpp"
#include "TransformerBlock.hpp"
#include "Tokenizer.hpp"
#include "PhysicsConfig.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <numeric>    // std::iota (nucleus sampling)
#include <algorithm>  // std::sort (nucleus sampling)
#include <random>     // std::mt19937, std::discrete_distribution

// ── RoPE: Rotary Positional Encoding (Feynman Phase Rotation) ─
// Physics: each token position = phase rotation in complex plane
//   For dimension pair (2i, 2i+1):
//     θ_i   = pos / 10000^(2i/d_model)   [frequency — Feynman phase]
//     x'[2i]   = x[2i]  * cos(θ_i) - x[2i+1] * sin(θ_i)
//     x'[2i+1] = x[2i]  * sin(θ_i) + x[2i+1] * cos(θ_i)
//
//   Relative position: RoPE encodes RELATIVE distance between tokens
//   naturally — Q·K dot product only depends on (pos_q - pos_k),
//   not absolute positions. This is the key advantage over learned PE.
//
//   In-place: applies rotation directly to X (seq, d_model)
inline void apply_rope(Tensor& X, int seq_len, int d_model) {
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < d_model / 2; ++i) {
            float theta = (float)pos /
                          std::pow(10000.0f, 2.0f * i / (float)d_model);
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);

            float x0 = X.at(pos, 2 * i);
            float x1 = X.at(pos, 2 * i + 1);

            X.at(pos, 2 * i)     = x0 * cos_t - x1 * sin_t;
            X.at(pos, 2 * i + 1) = x0 * sin_t + x1 * cos_t;
        }
    }
}

// ── ModelConfig — Single Source of Truth ─────────────────────
// BUG 5 FIX: One definition, consistent defaults everywhere.
// To scale up: pass explicit cfg to LOGOSModel constructor.
// Example: cfg.d_model=256; cfg.num_heads=8; cfg.num_layers=6; cfg.max_seq_len=512;
struct ModelConfig {
    int vocab_size  = 4096;   // default vocab (overridden by tokenizer in practice)
    int d_model     = 128;    // embedding dimension
    int num_heads   = 4;      // attention heads (d_model must be divisible by this)
    int num_layers  = 4;      // transformer blocks
    int max_seq_len = 128;    // max context length
};

class LOGOSModel {
public:
    ModelConfig cfg;
    Tensor embedding, lm_head;
    // pos_embedding REMOVED — replaced by RoPE (no learned PE parameters)
    // RoPE is parameter-free: positions encoded via phase rotation (apply_rope)
    std::vector<TransformerBlock> layers;

    // [WIRED] phys controls which previously-dead physics/vedic code paths
    // (ReynoldsBatchNorm, NavierStokesAttention, FeynmanDropout) actually
    // run in this model's layers. Kept OUT of ModelConfig on purpose — see
    // PhysicsConfig.hpp for why (checkpoint binary-layout compatibility).
    PhysicsConfig phys_cfg;

    LOGOSModel(const ModelConfig& config = {}, const PhysicsConfig& phys = {})
        : cfg(config), phys_cfg(phys),
          embedding({cfg.vocab_size, cfg.d_model}),
          lm_head  ({cfg.d_model,    cfg.vocab_size})
    {
        float scale = std::sqrt(2.0f / cfg.d_model);
        embedding.fill_random(-scale, scale);
        lm_head.fill_random(-scale, scale);
        layers.reserve(cfg.num_layers);
        for (int l = 0; l < cfg.num_layers; ++l)
            layers.emplace_back(cfg.d_model, cfg.num_heads, phys_cfg);
        std::cout << "LOGOS Model ready | layers=" << cfg.num_layers
                  << " d_model=" << cfg.d_model
                  << " vocab=" << cfg.vocab_size
                  << " | pos=RoPE"
                  << " | reynolds_norm="   << (phys_cfg.use_reynolds_norm   ? "on" : "off")
                  << " navier_stokes="     << (phys_cfg.use_navier_stokes   ? "on" : "off")
                  << " feynman_dropout="   << (phys_cfg.use_feynman_dropout ? "on" : "off")
                  << "\n";
    }

    // ── forward_with_hidden ───────────────────────────────────
    // IMPROVEMENT: Returns {logits, hidden_state} in ONE pass.
    // Eliminates the double-forward-pass bug in main.cpp training loop.
    // Use this in the training loop instead of calling forward() + recomputing
    // embedding+layers separately for gradient computation.
    //
    // Before (main.cpp bug): model.forward() for loss → rerun embedding+layers for grad
    // After: forward_with_hidden() → both logits AND hidden state from 1 pass
    //
    // Returns:
    //   first  = logits  {seq, vocab}  — for loss computation
    //   second = X_final {seq, d_model} — transformer output, for gradient
    std::pair<Tensor, Tensor> forward_with_hidden(const std::vector<int>& token_ids,
                                                   bool is_training = true) {
        int seq = (int)token_ids.size();
        if (seq > cfg.max_seq_len)
            throw std::runtime_error("Input too long: " + std::to_string(seq)
                                     + " > max_seq_len " + std::to_string(cfg.max_seq_len));

        Tensor X({seq, cfg.d_model}, 0.0f);
        for (int i = 0; i < seq; ++i) {
            int tok = std::max(0, std::min(token_ids[i], cfg.vocab_size-1));
            for (int j = 0; j < cfg.d_model; ++j)
                X.at(i,j) = embedding.at(tok,j);
        }
        apply_rope(X, seq, cfg.d_model);
        for (auto& block : layers) X = block.forward(X, is_training);

        Tensor hidden = X;  // save hidden state (copy, same shape)
        Tensor logits = vedic_gemm(X, lm_head);
        return {std::move(logits), std::move(hidden)};
    }

    // is_training: true during the training loop (enables ReynoldsBatchNorm's
    // running-stat EMA update and FeynmanDropout sampling). Pass false for
    // eval/generation so both behave deterministically (no dropout, uses
    // running BN stats once available).
    Tensor forward(const std::vector<int>& token_ids, bool is_training = true) {
        int seq = (int)token_ids.size();
        if (seq > cfg.max_seq_len)
            throw std::runtime_error("Input too long: " + std::to_string(seq)
                                     + " > max_seq_len " + std::to_string(cfg.max_seq_len));

        // Token embedding lookup (no additive pos_embedding anymore)
        Tensor X({seq, cfg.d_model}, 0.0f);
        for (int i = 0; i < seq; ++i) {
            int tok = std::max(0, std::min(token_ids[i], cfg.vocab_size-1));
            for (int j = 0; j < cfg.d_model; ++j)
                X.at(i,j) = embedding.at(tok,j);
        }

        // RoPE: Feynman phase rotation — encodes relative positions
        // Applied ONCE after embedding, before transformer blocks
        apply_rope(X, seq, cfg.d_model);

        for (auto& block : layers) X = block.forward(X, is_training);
        return vedic_gemm(X, lm_head);
    }

    // IMPROVEMENT: generate() now supports nucleus (top-p) sampling.
    // Before: greedy argmax only (temp was accepted but ignored for selection)
    // After:
    //   top_p < 1.0 → nucleus sampling (diverse, natural text)
    //   top_p = 1.0 → full temperature sampling
    //   temp  = 0.0 → greedy argmax (deterministic, same as before)
    //
    // Nucleus sampling (Holtzman et al. 2020):
    //   Sort vocab by probability descending.
    //   Keep smallest set S such that Σ_{v∈S} p(v) >= top_p.
    //   Sample uniformly within S (redistributed).
    //   This avoids both: low-entropy greedy (repetition) and
    //                     high-entropy full sampling (incoherence).
    std::vector<int> generate(std::vector<int> prompt,
                              int max_new = 50, float temp = 1.0f,
                              float top_p = 0.9f) {
        auto out = prompt;
        std::mt19937 gen(std::random_device{}());

        for (int s = 0; s < max_new; ++s) {
            std::vector<int> ctx = out;
            if ((int)ctx.size() > cfg.max_seq_len)
                ctx = {ctx.end() - cfg.max_seq_len, ctx.end()};

            Tensor logits = forward(ctx, /*is_training=*/false);
            int last = logits.rows()-1, vocab = logits.cols();

            // Greedy path (temp == 0 or very small)
            if (temp < 1e-5f) {
                int next = 0; float best = logits.at(last, 0);
                for (int v = 1; v < vocab; ++v)
                    if (logits.at(last, v) > best) { best = logits.at(last, v); next = v; }
                out.push_back(next);
                if (next == TOKEN_EOS) break;
                continue;
            }

            // Temperature-scaled softmax
            float max_l = logits.at(last, 0);
            for (int v = 1; v < vocab; ++v) max_l = std::max(max_l, logits.at(last, v));
            float sum = 0.0f;
            std::vector<float> p(vocab);
            for (int v = 0; v < vocab; ++v) {
                p[v] = std::exp((logits.at(last, v) - max_l) / temp);
                sum += p[v];
            }
            for (int v = 0; v < vocab; ++v) p[v] /= (sum + 1e-9f);

            // Nucleus (top-p) filtering
            int next = 0;
            if (top_p < 1.0f - 1e-5f) {
                // Sort indices by probability descending
                std::vector<int> idx(vocab);
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(),
                          [&](int a, int b){ return p[a] > p[b]; });

                // Find nucleus boundary
                float cumsum = 0.0f; int nucleus_end = 0;
                for (int i = 0; i < vocab; ++i) {
                    cumsum += p[idx[i]];
                    nucleus_end = i;
                    if (cumsum >= top_p) break;
                }

                // Renormalize within nucleus
                float nucleus_sum = 0.0f;
                for (int i = 0; i <= nucleus_end; ++i) nucleus_sum += p[idx[i]];
                std::vector<float> nucleus_p(nucleus_end + 1);
                for (int i = 0; i <= nucleus_end; ++i)
                    nucleus_p[i] = p[idx[i]] / (nucleus_sum + 1e-9f);

                // Sample from nucleus
                std::discrete_distribution<int> dist(nucleus_p.begin(), nucleus_p.end());
                next = idx[dist(gen)];
            } else {
                // Full distribution sampling
                std::discrete_distribution<int> dist(p.begin(), p.end());
                next = dist(gen);
            }

            out.push_back(next);
            if (next == TOKEN_EOS) break;
        }
        return out;
    }

    std::vector<Tensor*> parameters() {
        // pos_embedding removed (RoPE is parameter-free)
        std::vector<Tensor*> p = {&embedding, &lm_head};
        for (auto& l : layers) for (auto* w : l.parameters()) p.push_back(w);
        return p;
    }

    // Parameter count (approximate)
    long long count_parameters() const {
        long long total = embedding.total_size + lm_head.total_size; // no pos_embedding
        for (const auto& l : layers)
            total += (long long)cfg.d_model * cfg.d_model * 8   // attention matrices
                   + (long long)cfg.d_model * 4 * cfg.d_model * 2 // FFN
                   + cfg.d_model * 4;                             // LayerNorm
        return total;
    }
};
