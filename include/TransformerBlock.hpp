#pragma once
// ============================================================
//  LOGOS — TransformerBlock.hpp
// ============================================================
#include "Tensor.hpp"
#include "Attention.hpp"
#include "FeedForward.hpp"
#include "LayerNorm.hpp"

struct TransformerBlock {
    MultiHeadAttention mha;
    FeedForward        ffn;
    LayerNorm          ln1, ln2;

    TransformerBlock(int d_model_, int num_heads_)
        : mha(d_model_, num_heads_),
          ffn(d_model_),
          ln1(d_model_),
          ln2(d_model_)
    {}

    Tensor forward(const Tensor& X) {
        Tensor normed1  = ln1.forward(X);
        Tensor attn_out = mha.forward(normed1);
        Tensor h        = X + attn_out;
        Tensor normed2  = ln2.forward(h);
        Tensor ffn_out  = ffn.forward(normed2);
        return h + ffn_out;
    }

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> p;
        auto app = [&](std::vector<Tensor*> v){
            p.insert(p.end(), v.begin(), v.end());
        };
        app(mha.parameters());
        app(ffn.parameters());
        app(ln1.parameters());
        app(ln2.parameters());
        return p;
    }
};
