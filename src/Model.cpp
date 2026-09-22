// ============================================================
//  LOGOS — Model.cpp
//  Complete LLM = Embedding + N×TransformerBlock + LM Head
//
//  Config defaults (10M parameters target):
//    d_model   = 256
//    num_heads = 8
//    num_layers= 6
//    vocab_size= 4096
//    max_seq   = 512
//
//  Parameter count estimate:
//    Embedding: 4096 × 256          = ~1.0M
//    Each block: ~256² × 12         = ~0.8M × 6 = ~4.8M
//    LM Head: 256 × 4096            = ~1.0M
//    Total: ~7-10M ✅
// ============================================================
#ifndef LOGOS_MODEL_CPP
#define LOGOS_MODEL_CPP

#include "../include/Tensor.hpp"
#include "TransformerBlock.cpp"
#include "VedicGEMM.cpp"
#include <fstream>
#include <iostream>

struct ModelConfig {
    int vocab_size  = 4096;
    int d_model     = 256;
    int num_heads   = 8;
    int num_layers  = 6;
    int max_seq_len = 512;
};

class LOGOSModel {
public:
    ModelConfig cfg;
    Tensor embedding;       // (vocab_size, d_model) — word embeddings
    Tensor pos_embedding;   // (max_seq, d_model) — positional embeddings
    std::vector<TransformerBlock> layers;
    Tensor lm_head;         // (d_model, vocab_size) — output projection

    LOGOSModel(const ModelConfig& config = {})
        : cfg(config),
          embedding({cfg.vocab_size, cfg.d_model}),
          pos_embedding({cfg.max_seq_len, cfg.d_model}),
          lm_head({cfg.d_model, cfg.vocab_size})
    {
        float scale = std::sqrt(2.0f / cfg.d_model);
        embedding.fill_random(-scale, scale);
        pos_embedding.fill_random(-0.01f, 0.01f);
        lm_head.fill_random(-scale, scale);

        layers.reserve(cfg.num_layers);
        for (int l = 0; l < cfg.num_layers; ++l)
            layers.emplace_back(cfg.d_model, cfg.num_heads);

        std::cout << "✅ LOGOS Model initialized\n";
        std::cout << "   Layers: " << cfg.num_layers
                  << " | d_model: " << cfg.d_model
                  << " | heads: " << cfg.num_heads
                  << " | vocab: " << cfg.vocab_size << "\n";
        std::cout << "   Approx params: ~"
                  << count_parameters() / 1000000 << "M\n";
    }

    // ── Forward pass: token ids → logits ─────────────────────
    // token_ids: vector of int (length = seq_len)
    // Returns: Tensor (seq_len, vocab_size) — raw logits
    Tensor forward(const std::vector<int>& token_ids) {
        int seq = static_cast<int>(token_ids.size());
        if (seq > cfg.max_seq_len) {
            throw std::runtime_error("Input too long: " + std::to_string(seq) +
                                     " > max_seq_len " + std::to_string(cfg.max_seq_len));
        }

        // Step 1: Embedding lookup (Vedic GEMM via one-hot style)
        // More efficient: direct index into embedding rows
        Tensor X({seq, cfg.d_model}, 0.0f);
        for (int i = 0; i < seq; ++i) {
            int tok = token_ids[i];
            if (tok < 0 || tok >= cfg.vocab_size) tok = 0;  // UNK
            for (int j = 0; j < cfg.d_model; ++j)
                X.at(i, j) = embedding.at(tok, j) + pos_embedding.at(i, j);
        }

        // Step 2: N Transformer blocks
        for (auto& block : layers)
            X = block.forward(X);

        // Step 3: LM Head → logits (Vedic GEMM)
        return vedic_gemm(X, lm_head);   // (seq, vocab_size)
    }

    // ── Greedy text generation ────────────────────────────────
    std::vector<int> generate(std::vector<int> prompt,
                              int max_new_tokens = 50,
                              float temperature = 1.0f)
    {
        std::vector<int> generated = prompt;
        for (int step = 0; step < max_new_tokens; ++step) {
            // Truncate to max context
            std::vector<int> context = generated;
            if ((int)context.size() > cfg.max_seq_len)
                context = std::vector<int>(context.end() - cfg.max_seq_len, context.end());

            Tensor logits = forward(context);  // (seq, vocab)

            // Last token logits → next token
            int last_row = logits.rows() - 1;
            int vocab = logits.cols();

            // Apply temperature + softmax → greedy argmax
            float max_logit = logits.at(last_row, 0);
            for (int v = 1; v < vocab; ++v)
                max_logit = std::max(max_logit, logits.at(last_row, v));

            float sum = 0.0f;
            std::vector<float> probs(vocab);
            for (int v = 0; v < vocab; ++v) {
                probs[v] = std::exp((logits.at(last_row, v) - max_logit) / temperature);
                sum += probs[v];
            }

            // Greedy: pick max probability token
            int next_tok = 0;
            float best = 0.0f;
            for (int v = 0; v < vocab; ++v)
                if (probs[v]/sum > best) { best = probs[v]/sum; next_tok = v; }

            generated.push_back(next_tok);
            if (next_tok == 2) break;  // EOS token
        }
        return generated;
    }

    // ── Parameter count ───────────────────────────────────────
    long long count_parameters() const {
        long long total = 0;
        total += embedding.total_size;
        total += pos_embedding.total_size;
        total += lm_head.total_size;
        // Each block: 4 heads × 4 matrices + FFN + LayerNorms
        // Approximate
        total += (long long)cfg.num_layers *
                 (4LL * cfg.num_heads * cfg.d_model * (cfg.d_model/cfg.num_heads) * 4
                  + cfg.d_model * cfg.d_model    // W_proj
                  + cfg.d_model * 4 * cfg.d_model * 2  // FFN
                  + cfg.d_model * 4);             // LayerNorm gammas/betas
        return total;
    }

    // ── Save/Load weights ─────────────────────────────────────
    bool save_weights(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        auto write_tensor = [&](const Tensor& t) {
            int n = t.total_size;
            f.write(reinterpret_cast<const char*>(&n), sizeof(int));
            f.write(reinterpret_cast<const char*>(t.data.data()), n * sizeof(float));
        };

        write_tensor(embedding);
        write_tensor(pos_embedding);
        write_tensor(lm_head);
        // Note: layer weights save in Checkpoint.cpp (full)
        std::cout << "✅ Weights saved to " << path << "\n";
        return true;
    }

    // ── All parameters (for optimizer) ───────────────────────
    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> params = {&embedding, &pos_embedding, &lm_head};
        for (auto& layer : layers)
            for (auto* p : layer.parameters())
                params.push_back(p);
        return params;
    }
};

#endif
