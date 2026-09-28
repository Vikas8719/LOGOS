#pragma once
// ============================================================
//  LOGOS — PhysicsConfig.hpp   [NEW — wiring pass]
//
struct PhysicsConfig {
    // ── Reynolds-adaptive normalisation (LayerNorm.hpp) ──────
    // true  → ReynoldsBatchNorm (laminar/turbulent blended norm)
    // false → collapses to plain LayerNorm behaviour (Re_crit pinned huge)
    bool  use_reynolds_norm   = true;

    // ── Navier-Stokes viscous-advective attention (Attention.hpp) ──
    bool  use_navier_stokes   = true;
    float ns_alpha            = 0.3f;   // blend: 0=pure standard, 1=pure NS
    float ns_eta              = 0.1f;   // advection strength
    float ns_nu               = 0.05f;  // kinematic viscosity

    // ── Shunyam sparse attention (Attention.hpp) ─────────────
    // (was already wired via AttentionHead's own default, kept
    //  configurable here so it's controlled from one place)
    bool  use_sparse_attn     = true;
    int   sparse_window       = 8;
    int   sparse_stride       = 4;

    // ── Feynman path-integral dropout (FeedForward.hpp) ──────
    bool  use_feynman_dropout = true;
    float dropout_p           = 0.1f;
    float dropout_hbar        = 1.0f;
};
