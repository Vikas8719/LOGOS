#pragma once
// ============================================================
//  LOGOS — Model.hpp
//
//  BUG 5 FIX: ModelConfig defaults unified — .hpp aur .cpp mismatch band
//    Pehle .hpp:  d_model=128, num_heads=4, num_layers=4, max_seq_len=128
//    Pehle .cpp:  d_model=256, num_heads=8, num_layers=6, max_seq_len=512
//    → koi bhi ModelConfig{} use kare toh alag values milti thi
//
//    Ab: SINGLE SOURCE OF TRUTH yahan (Model.hpp) hai.
//        Model.cpp mein duplicate ModelConfig definition remove kar di gayi.
//        Chosen defaults: d_model=128, heads=4, layers=4, seq=128
//        (conservative — fast iteration, easily scalable via explicit cfg)
//
//  RoPE UPGRADE: pos_embedding (learned, additive) → RoPE (parameter-free)
//    Feynman Phase Rotation: x'[2i], x'[2i+1] = rotate by θ_i = pos/10000^(2i/d)
//    Benefit: encodes RELATIVE positions naturally, better generalisation
//    apply_rope() defined above LOGOSModel — called once in forward()
//
//  ✅ ModelConfig ONLY defined here — Model.cpp mein nahi
//  ✅ LOGOSModel ONLY defined here — Model.cpp is now DEAD (not compiled)
//  ✅ RoPE replaces learned pos_embedding — parameter-free, physics-grounded
// ============================================================
#include "Tensor.hpp"
#include "VedicGEMM.hpp"
#include "TransformerBlock.hpp"
#include "Tokenizer.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>

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

    LOGOSModel(const ModelConfig& config = {})
        : cfg(config),
          embedding({cfg.vocab_size, cfg.d_model}),
          lm_head  ({cfg.d_model,    cfg.vocab_size})
    {
        float scale = std::sqrt(2.0f / cfg.d_model);
        embedding.fill_random(-scale, scale);
        lm_head.fill_random(-scale, scale);
        layers.reserve(cfg.num_layers);
        for (int l = 0; l < cfg.num_layers; ++l)
            layers.emplace_back(cfg.d_model, cfg.num_heads);
        std::cout << "LOGOS Model ready | layers=" << cfg.num_layers
                  << " d_model=" << cfg.d_model
                  << " vocab=" << cfg.vocab_size
                  << " | pos=RoPE\n";
    }

    Tensor forward(const std::vector<int>& token_ids) {
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

        for (auto& block : layers) X = block.forward(X);
        return vedic_gemm(X, lm_head);
    }

    std::vector<int> generate(std::vector<int> prompt,
                              int max_new = 50, float temp = 1.0f) {
        auto out = prompt;
        for (int s = 0; s < max_new; ++s) {
            std::vector<int> ctx = out;
            if ((int)ctx.size() > cfg.max_seq_len)
                ctx = {ctx.end() - cfg.max_seq_len, ctx.end()};
            Tensor logits = forward(ctx);
            int last = logits.rows()-1, vocab = logits.cols();
            float max_l = logits.at(last,0);
            for (int v=1;v<vocab;++v) max_l = std::max(max_l, logits.at(last,v));
            float sum=0; std::vector<float> p(vocab);
            for (int v=0;v<vocab;++v){ p[v]=std::exp((logits.at(last,v)-max_l)/temp); sum+=p[v]; }
            int next=0; float best=0;
            for (int v=0;v<vocab;++v) if(p[v]/sum>best){best=p[v]/sum;next=v;}
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
