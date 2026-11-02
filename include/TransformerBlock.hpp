#pragma once
#include "Tensor.hpp"
#include "Attention.hpp"
#include "FeedForward.hpp"
#include "LayerNorm.hpp"
#include "PhysicsConfig.hpp"

inline Tensor divergence_free(const Tensor& U) {
    int seq = U.rows(), dim = U.cols();
    Tensor U2(U.shape);
    for (int i = 0; i < seq; ++i) {
        float mean = 0.0f;
        for (int j = 0; j < dim; ++j) mean += U.at(i, j);
        mean /= (float)dim;
        for (int j = 0; j < dim; ++j)
            U2.at(i, j) = U.at(i, j) - mean;  // u' = u - mean(u)
    }
    return U2;
}

struct TransformerBlock {
    MultiHeadAttention mha;
    FeedForward        ffn;
    ReynoldsBatchNorm  ln1, ln2;

    TransformerBlock(int d_model_, int num_heads_, const PhysicsConfig& phys = {})
        : mha(d_model_, num_heads_),
          ffn(d_model_, 0, phys.dropout_p, phys.dropout_hbar, phys.use_feynman_dropout),
          ln1(d_model_),
          ln2(d_model_)
    {
        for (auto& h : mha.heads) {
            h.use_sparse        = phys.use_sparse_attn;
            h.sparse_window     = phys.sparse_window;
            h.sparse_stride     = phys.sparse_stride;
            h.use_navier_stokes = phys.use_navier_stokes;
            h.ns_alpha          = phys.ns_alpha;
            h.ns_eta            = phys.ns_eta;
            h.ns_nu             = phys.ns_nu;
        }
        if (!phys.use_reynolds_norm) {
            ln1.Re_crit = 1e6f;
            ln2.Re_crit = 1e6f;
        }
    }

    Tensor forward(const Tensor& X, bool is_training = true) {
        Tensor normed1  = ln1.forward(X, is_training);
        Tensor attn_out = mha.forward(normed1);
        Tensor h        = divergence_free(X + attn_out);

        Tensor normed2  = ln2.forward(h, is_training);
        Tensor ffn_out  = ffn.forward(normed2, is_training);
        return divergence_free(h + ffn_out);
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
