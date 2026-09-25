#pragma once
// ============================================================
//  LOGOS — TransformerBlock.hpp
//  Residual connections: Incompressible Flow (∇·u = 0)
//
//  Physics: residual stream = incompressible fluid velocity field
//    ∇·u = 0 means no "sources" or "sinks" — information is
//    conserved, not created or destroyed in residual path.
//
//  Implementation: after each residual add, we project out the
//    "compressible component" by mean-centering across d_model.
//    This enforces ∂u_i/∂x_i = 0 (discrete divergence = 0).
//
//    Divergence-free projection:  u' = u - mean(u)
//    → sum(u') = 0 over feature dimension → ∇·u ≈ 0
//    → residual stream has zero net "flux" per token position
// ============================================================
#include "Tensor.hpp"
#include "Attention.hpp"
#include "FeedForward.hpp"
#include "LayerNorm.hpp"
#include <numeric>

// ── Divergence-free projection (∇·u = 0) ─────────────────────
// Subtracts per-row mean → zero divergence in feature space
// Applied after each residual add to enforce incompressibility
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
        // Residual 1: incompressible flow — project to ∇·u = 0
        Tensor h        = divergence_free(X + attn_out);

        Tensor normed2  = ln2.forward(h);
        Tensor ffn_out  = ffn.forward(normed2);
        // Residual 2: incompressible flow — project to ∇·u = 0
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
