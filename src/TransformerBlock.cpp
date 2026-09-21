// ============================================================
//  LOGOS — TransformerBlock.cpp
//  One Transformer Layer = Attention + FFN + 2× LayerNorm
//
//  Flow: X → [LN1] → [MHA] → (+X) → [LN2] → [FFN] → (+X)
//  Residual connections: gradient flow stable rehta hai
// ============================================================
#include "../include/Tensor.hpp"
#include "Attention.cpp"
#include "FeedForward.cpp"
#include "LayerNorm.cpp"

struct TransformerBlock {
    MultiHeadAttention mha;
    FeedForward        ffn;
    LayerNorm          ln1, ln2;
    int d_model;

    TransformerBlock(int d_model_, int num_heads_)
        : mha(d_model_, num_heads_),
          ffn(d_model_),
          ln1(d_model_),
          ln2(d_model_),
          d_model(d_model_)
    {}

    // X: (seq_len, d_model) → (seq_len, d_model)
    Tensor forward(const Tensor& X) {
        // Pre-LN attention path
        Tensor normed1 = ln1.forward(X);
        Tensor attn_out = mha.forward(normed1);
        // Residual connection 1
        Tensor h = X + attn_out;

        // Pre-LN FFN path
        Tensor normed2 = ln2.forward(h);
        Tensor ffn_out = ffn.forward(normed2);
        // Residual connection 2
        return h + ffn_out;
    }

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> params;
        auto append = [&](std::vector<Tensor*> p){
            params.insert(params.end(), p.begin(), p.end());
        };
        append(mha.parameters());
        append(ffn.parameters());
        append(ln1.parameters());
        append(ln2.parameters());
        return params;
    }
};
