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
#include "PhysicsConfig.hpp"
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
    // [WIRED] was plain LayerNorm — ReynoldsBatchNorm (LayerNorm.hpp) was
    // defined but never instantiated anywhere. It's a superset behaviourally:
    // when PhysicsConfig::use_reynolds_norm is false we pin Re_crit huge so
    // laminar_weight()≈1 and it degenerates to plain LayerNorm exactly.
    ReynoldsBatchNorm  ln1, ln2;

    TransformerBlock(int d_model_, int num_heads_, const PhysicsConfig& phys = {})
        : mha(d_model_, num_heads_),
          ffn(d_model_, 0, phys.dropout_p, phys.dropout_hbar, phys.use_feynman_dropout),
          ln1(d_model_),
          ln2(d_model_)
    {
        // [WIRED] NavierStokesAttention + Shunyam sparse mask (Attention.hpp)
        // existed as toggleable fields on AttentionHead but nothing ever set
        // them from model config — use_navier_stokes defaulted to false, so
        // the NS code path never ran. Now driven by PhysicsConfig.
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

    // is_training gates: ReynoldsBatchNorm running-stat EMA updates,
    // and (via ffn.forward) FeynmanDropout's Bernoulli/Beta sampling.
    Tensor forward(const Tensor& X, bool is_training = true) {
        Tensor normed1  = ln1.forward(X, is_training);
        Tensor attn_out = mha.forward(normed1);
        // Residual 1: incompressible flow — project to ∇·u = 0
        Tensor h        = divergence_free(X + attn_out);

        Tensor normed2  = ln2.forward(h, is_training);
        Tensor ffn_out  = ffn.forward(normed2, is_training);
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
