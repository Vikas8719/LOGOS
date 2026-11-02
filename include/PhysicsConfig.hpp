#pragma once
struct PhysicsConfig {
    bool  use_reynolds_norm   = true;

    bool  use_navier_stokes   = true;
    float ns_alpha            = 0.3f;   // blend: 0=pure standard, 1=pure NS
    float ns_eta              = 0.1f;   // advection strength
    float ns_nu               = 0.05f;  // kinematic viscosity

    bool  use_sparse_attn     = true;
    int   sparse_window       = 8;
    int   sparse_stride       = 4;

    bool  use_feynman_dropout = true;
    float dropout_p           = 0.1f;
    float dropout_hbar        = 1.0f;
};
