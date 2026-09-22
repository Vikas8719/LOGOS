#pragma once
// ============================================================
//  LOGOS — Model.hpp
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

struct ModelConfig {
    int vocab_size  = 4096;
    int d_model     = 128;
    int num_heads   = 4;
    int num_layers  = 4;
    int max_seq_len = 128;
};

class LOGOSModel {
public:
    ModelConfig cfg;
    Tensor embedding, pos_embedding, lm_head;
    std::vector<TransformerBlock> layers;

    LOGOSModel(const ModelConfig& config = {})
        : cfg(config),
          embedding    ({cfg.vocab_size,  cfg.d_model}),
          pos_embedding({cfg.max_seq_len, cfg.d_model}),
          lm_head      ({cfg.d_model,     cfg.vocab_size})
    {
        float scale = std::sqrt(2.0f / cfg.d_model);
        embedding.fill_random(-scale, scale);
        pos_embedding.fill_random(-0.01f, 0.01f);
        lm_head.fill_random(-scale, scale);
        layers.reserve(cfg.num_layers);
        for (int l = 0; l < cfg.num_layers; ++l)
            layers.emplace_back(cfg.d_model, cfg.num_heads);
        std::cout << "LOGOS Model ready | layers=" << cfg.num_layers
                  << " d_model=" << cfg.d_model
                  << " vocab=" << cfg.vocab_size << "\n";
    }

    Tensor forward(const std::vector<int>& token_ids) {
        int seq = (int)token_ids.size();
        if (seq > cfg.max_seq_len)
            throw std::runtime_error("Input too long");

        Tensor X({seq, cfg.d_model}, 0.0f);
        for (int i = 0; i < seq; ++i) {
            int tok = std::max(0, std::min(token_ids[i], cfg.vocab_size-1));
            for (int j = 0; j < cfg.d_model; ++j)
                X.at(i,j) = embedding.at(tok,j) + pos_embedding.at(i,j);
        }
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
        std::vector<Tensor*> p = {&embedding, &pos_embedding, &lm_head};
        for (auto& l : layers) for (auto* w : l.parameters()) p.push_back(w);
        return p;
    }
};
