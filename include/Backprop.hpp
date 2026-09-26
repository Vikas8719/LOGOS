#pragma once
// ============================================================
//  LOGOS — Backprop.hpp
//  Analytical Gradient Computation (no autograd engine needed)
//
//  FIX: Previously all transformer layer weights (grads[2+]) received
//  only a scalar proxy signal (dX_norm * 0.1) — not real gradients.
//  This meant layers learned entirely via Langevin thermal noise,
//  with no gradient direction at all. Training worked (Langevin can
//  learn without gradients) but was dramatically slower than it needed
//  to be and would never converge well on real tasks.
//
//  This file implements ANALYTICAL backprop for every layer type
//  in LOGOS without requiring a full autograd engine:
//
//  1. GEMM backward        : dA = dC @ B^T,  dB = A^T @ dC
//  2. GELU backward        : dX = dY * gelu'(X)
//  3. Bias backward        : db = sum(dY, axis=0)
//  4. LayerNorm backward   : standard LN gradient (Ba et al.)
//  5. Softmax-CE backward  : dLogits = softmax(logits) - one_hot(target)
//  6. FFN full backward    : chain rule through W2→GELU→W1
//  7. Attention backward   : dQ,dK,dV,dW_O via chain through softmax
//  8. TransformerBlock     : residual + divergence-free projection grad
//
//  All gradients ACCUMULATE into existing grad tensors (+=, not =).
//  Call before clip_gradients() + optimizer step.
//
//  Physics note: Langevin thermal noise still runs on top of these
//  real gradients via HybridSHMOptimizer — they are COMPLEMENTARY.
//  Real gradients give direction; Langevin gives exploration + flat
//  minima preference (better generalisation, Keskar et al. 2017).
// ============================================================
#include "Tensor.hpp"
#include "VedicGEMM.hpp"
#include "Attention.hpp"
#include "FeedForward.hpp"
#include "LayerNorm.hpp"
#include "TransformerBlock.hpp"
#include "Model.hpp"
#include <cmath>
#include <vector>

// ── 1. GEMM backward ─────────────────────────────────────────
// Forward:  C = A @ B
// Backward: dA += dC @ B^T
//           dB += A^T @ dC
// Both accumulate (+=) so multiple token positions sum correctly.
inline void gemm_backward(const Tensor& A,   // (M, K) — forward input
                           const Tensor& B,   // (K, N) — weight
                           const Tensor& dC,  // (M, N) — upstream gradient
                           Tensor& dA,        // (M, K) — grad w.r.t. input
                           Tensor& dB)        // (K, N) — grad w.r.t. weight
{
    // dA += dC @ B^T
    Tensor B_T = B.transpose();   // (N, K)
    Tensor dA_contrib = vedic_gemm(dC, B_T);  // (M, K)
    for (int i = 0; i < dA.total_size; ++i) dA.data[i] += dA_contrib.data[i];

    // dB += A^T @ dC
    Tensor A_T = A.transpose();   // (K, M)
    Tensor dB_contrib = vedic_gemm(A_T, dC);  // (K, N)
    for (int i = 0; i < dB.total_size; ++i) dB.data[i] += dB_contrib.data[i];
}

// ── 2. GELU backward ─────────────────────────────────────────
// Forward:  y = GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
// Backward: dx = dy * GELU'(x)
// GELU'(x) = 0.5*tanh(c*(x+0.044715x³))
//           + 0.5*x*sech²(c*(x+0.044715x³))*(c+3*0.044715*c*x²)
// where c = 0.7978845608 (= sqrt(2/pi))
inline Tensor gelu_backward(const Tensor& X,  // (M, N) — pre-activation
                              const Tensor& dY) // (M, N) — upstream
{
    constexpr float c  = 0.7978845608f;
    constexpr float a  = 0.044715f;
    Tensor dX(X.shape, 0.0f);
    for (int i = 0; i < X.total_size; ++i) {
        float x  = X.data[i];
        float x3 = x * x * x;
        float t  = std::tanh(c * (x + a * x3));
        float dt = 1.0f - t * t;  // sech^2
        // GELU'(x) = 0.5*(1+t) + 0.5*x*dt*(c*(1 + 3*a*x^2))
        float dgelu = 0.5f * (1.0f + t)
                    + 0.5f * x * dt * (c * (1.0f + 3.0f * a * x * x));
        dX.data[i] = dY.data[i] * dgelu;
    }
    return dX;
}

// ── 3. Bias backward ─────────────────────────────────────────
// Forward:  Y = X + bias  (bias broadcast over rows)
// Backward: dbias[j] = sum_i dY[i,j]
inline Tensor bias_backward(const Tensor& dY)  // (seq, dim)
{
    int seq = dY.rows(), dim = dY.cols();
    Tensor dbias({1, dim}, 0.0f);
    for (int i = 0; i < seq; ++i)
        for (int j = 0; j < dim; ++j)
            dbias.at(0, j) += dY.at(i, j);
    return dbias;
}

// ── 4. LayerNorm backward ────────────────────────────────────
// Forward:  y_i = gamma * (x_i - mean) / std + beta
// Backward: standard LN gradient (Eq. 4, Ba et al. 2016)
//
// For one token row of length D:
//   d_xhat[i] = dy[i] * gamma[i]
//   d_var  = sum_i( d_xhat[i] * (x[i]-mean) ) * (-0.5) * (var+eps)^{-3/2}
//   d_mean = sum_i(-d_xhat[i]/std) + d_var * sum_i(-2*(x[i]-mean)/D)
//   dx[i]  = d_xhat[i]/std + d_var*2*(x[i]-mean)/D + d_mean/D
//
// gamma and beta grads also accumulated.
inline void layernorm_backward(const Tensor&  X,      // (seq, D) — input
                                const Tensor&  dY,     // (seq, D) — upstream
                                const Tensor&  gamma,  // (1, D) — scale
                                float          eps,
                                Tensor& dX,            // (seq, D) — output grad
                                Tensor& dgamma,        // (1, D)
                                Tensor& dbeta)         // (1, D)
{
    int seq = X.rows(), D = X.cols();
    for (int i = 0; i < seq; ++i) {
        // Recompute mean + variance (needed for grad; same as forward)
        float mean = 0.0f, M2 = 0.0f;
        for (int j = 0; j < D; ++j) {
            float delta = X.at(i,j) - mean;
            mean += delta / (j + 1);
            M2   += delta * (X.at(i,j) - mean);
        }
        float var   = M2 / D;
        float std_  = std::sqrt(var + eps);
        float inv_s = 1.0f / std_;

        // d_xhat = dY * gamma (element-wise)
        // dbeta += dY;  dgamma += dY * xhat
        float d_var  = 0.0f, d_mean = 0.0f;
        std::vector<float> d_xhat(D);
        for (int j = 0; j < D; ++j) {
            float xhat = (X.at(i,j) - mean) * inv_s;
            d_xhat[j]  = dY.at(i,j) * gamma[j];
            dbeta.at(0,j)  += dY.at(i,j);
            dgamma.at(0,j) += dY.at(i,j) * xhat;

            d_var  += d_xhat[j] * (X.at(i,j) - mean);
        }
        d_var  *= -0.5f * inv_s * inv_s * inv_s;
        for (int j = 0; j < D; ++j)
            d_mean += -d_xhat[j] * inv_s
                     + d_var * (-2.0f * (X.at(i,j) - mean) / D);

        for (int j = 0; j < D; ++j)
            dX.at(i,j) += d_xhat[j] * inv_s
                        + d_var * 2.0f * (X.at(i,j) - mean) / D
                        + d_mean / D;
    }
}

// ── 5. Softmax + Cross-Entropy backward (fused) ──────────────
// Forward:  loss = -log softmax(logits)[target]
// Backward: dlogits[v] = softmax(logits)[v] - 1{v==target}
// Averaged over seq (divides by seq for normalisation consistency).
// Returns dLogits (seq, vocab) — pass directly to lm_head backward.
inline Tensor softmax_ce_backward(const Tensor&         logits,   // (seq, vocab)
                                   const std::vector<int>& targets, // (seq,)
                                   float&                  loss_out) // scalar loss
{
    int seq = logits.rows(), vocab = logits.cols();
    Tensor dLogits(logits.shape, 0.0f);
    float total_loss = 0.0f;
    int   cnt = 0;

    for (int i = 0; i < seq && i < (int)targets.size(); ++i) {
        int tgt = targets[i];
        if (tgt < 0 || tgt >= vocab) continue;

        // Numerically stable softmax
        float mx = logits.at(i, 0);
        for (int v = 1; v < vocab; ++v) mx = std::max(mx, logits.at(i,v));
        float sm = 0.0f;
        for (int v = 0; v < vocab; ++v) sm += std::exp(logits.at(i,v) - mx);

        total_loss += -(logits.at(i,tgt) - mx - std::log(sm + 1e-10f));
        for (int v = 0; v < vocab; ++v) {
            float p = std::exp(logits.at(i,v) - mx) / (sm + 1e-10f);
            dLogits.at(i,v) = (p - (v == tgt ? 1.0f : 0.0f)) / (float)seq;
        }
        ++cnt;
    }

    loss_out = cnt > 0 ? total_loss / cnt : 0.0f;
    return dLogits;
}

// ── 6. FFN full backward ─────────────────────────────────────
// Forward:
//   H = GELU(X @ W1 + b1)        [pre-act stored for GELU grad]
//   Y = H @ W2 + b2
//
// Backward (chain rule):
//   dH  = dY @ W2^T
//   dW2 += H^T @ dY
//   db2 += sum(dY, axis=0)
//   dPre = dH * GELU'(pre_act)
//   dW1 += X^T @ dPre
//   db1 += sum(dPre, axis=0)
//   dX  += dPre @ W1^T
//
// pre_act must be provided (X @ W1 + b1 BEFORE GELU) — caller computes it.
inline Tensor ffn_backward(const FeedForward& ffn,
                            const Tensor& X,        // (seq, d_model)  — FFN input
                            const Tensor& pre_act,  // (seq, d_ff)     — pre-GELU
                            const Tensor& dY,       // (seq, d_model)  — upstream
                            Tensor& dW1, Tensor& db1,
                            Tensor& dW2, Tensor& db2)
{
    // ── dW2, db2 ──
    // dH = dY @ W2^T
    Tensor W2_T   = ffn.W2.transpose();
    Tensor dH     = vedic_gemm(dY, W2_T);      // (seq, d_ff)

    // dW2 += H^T @ dY (H = GELU(pre_act))
    Tensor H(pre_act.shape);
    for (int i = 0; i < pre_act.total_size; ++i)
        H.data[i] = gelu(pre_act.data[i]);
    Tensor H_T = H.transpose();
    Tensor dW2_c = vedic_gemm(H_T, dY);
    for (int i = 0; i < dW2.total_size; ++i) dW2.data[i] += dW2_c.data[i];

    // db2 += sum(dY)
    Tensor db2_c = bias_backward(dY);
    for (int i = 0; i < db2.total_size; ++i) db2.data[i] += db2_c.data[i];

    // ── dPre = dH * GELU'(pre_act) ──
    Tensor dPre = gelu_backward(pre_act, dH);   // (seq, d_ff)

    // ── dW1, db1 ──
    Tensor X_T  = X.transpose();
    Tensor dW1_c = vedic_gemm(X_T, dPre);
    for (int i = 0; i < dW1.total_size; ++i) dW1.data[i] += dW1_c.data[i];

    Tensor db1_c = bias_backward(dPre);
    for (int i = 0; i < db1.total_size; ++i) db1.data[i] += db1_c.data[i];

    // ── dX = dPre @ W1^T ──
    Tensor W1_T = ffn.W1.transpose();
    return vedic_gemm(dPre, W1_T);    // (seq, d_model)
}

// ── 7. Attention head backward ───────────────────────────────
// Forward (standard path):
//   Q  = X @ W_Q,  K = X @ W_K,  V = X @ W_V
//   S  = Q @ K^T / sqrt(d_k)  + mask
//   A  = softmax(S)              (Boltzmann weights)
//   O  = A @ V
//   Y  = O @ W_O
//
// Backward:
//   dY → dO  via W_O^T
//   dW_O   += O^T @ dY
//   dO     = dY @ W_O^T
//   dA     = dO @ V^T
//   dV     = A^T @ dO
//   dW_V   += X^T @ dV_pre  where dV_pre = dV (since V=X@W_V)
//   dS     = softmax_backward(A, dA)
//   scale  = 1/sqrt(d_k)
//   dQ     = dS * scale @ K
//   dK     = dS^T * scale @ Q
//   dW_Q   += X^T @ dQ
//   dW_K   += X^T @ dK
//   dX_head = dQ @ W_Q^T + dK @ W_K^T + dV_pre @ W_V^T
//
// Softmax backward:
//   Given A (softmax output) and dA (upstream):
//   dS[i,j] = A[i,j] * (dA[i,j] - sum_k(dA[i,k]*A[i,k]))
//   This is the Jacobian-vector product of softmax.
inline Tensor softmax_backward_2d(const Tensor& A,   // (seq, seq) softmax output
                                   const Tensor& dA)  // (seq, seq) upstream
{
    int rows = A.rows(), cols = A.cols();
    Tensor dS(A.shape, 0.0f);
    for (int i = 0; i < rows; ++i) {
        // dot = sum_k(dA[i,k] * A[i,k])
        float dot = 0.0f;
        for (int k = 0; k < cols; ++k) dot += dA.at(i,k) * A.at(i,k);
        for (int j = 0; j < cols; ++j)
            dS.at(i,j) = A.at(i,j) * (dA.at(i,j) - dot);
    }
    return dS;
}

inline Tensor attention_head_backward(
    AttentionHead& head,
    const Tensor& X,      // (seq, d_model) — block input to this head
    const Tensor& dY,     // (seq, d_model) — upstream gradient
    Tensor& dW_Q, Tensor& dW_K, Tensor& dW_V, Tensor& dW_O)
{
    float inv_sqrt_dk = 1.0f / std::sqrt((float)head.d_k);
    int   seq         = X.rows();

    // ── Recompute forward intermediates ──────────────────────
    // (same computation as AttentionHead::forward — needed for grads)
    Tensor Q = vedic_gemm(X, head.W_Q);  // (seq, d_k)
    Tensor K = vedic_gemm(X, head.W_K);  // (seq, d_k)
    Tensor V = vedic_gemm(X, head.W_V);  // (seq, d_k)

    // Select mask (same logic as forward)
    Tensor mask;
    if (head.use_sparse && seq > head.sparse_window)
        mask = shunyam_sparse_mask_cached(seq, head.sparse_window, head.sparse_stride);
    else
        mask = causal_mask_cached(seq);

    Tensor K_T = K.transpose();
    Tensor S   = vedic_gemm(Q, K_T);        // (seq, seq) — raw scores
    S += mask;
    float temperature = std::sqrt((float)head.d_k);
    Tensor A   = boltzmann_softmax(S, temperature);  // (seq, seq)
    Tensor O   = vedic_gemm(A, V);                   // (seq, d_k)

    // ── Backward through W_O ──────────────────────────────────
    // dO = dY @ W_O^T
    Tensor W_O_T = head.W_O.transpose();
    Tensor dO    = vedic_gemm(dY, W_O_T);        // (seq, d_k)

    // dW_O += O^T @ dY
    Tensor O_T   = O.transpose();
    Tensor dW_O_c = vedic_gemm(O_T, dY);         // (d_k, d_model)
    for (int i = 0; i < dW_O.total_size; ++i) dW_O.data[i] += dW_O_c.data[i];

    // ── Backward through attn weights ─────────────────────────
    // dA = dO @ V^T
    Tensor V_T = V.transpose();
    Tensor dA  = vedic_gemm(dO, V_T);            // (seq, seq)

    // dV = A^T @ dO
    Tensor A_T  = A.transpose();
    Tensor dV   = vedic_gemm(A_T, dO);           // (seq, d_k)

    // ── Backward through softmax ──────────────────────────────
    // dS = softmax_backward(A, dA)   (scaled by 1/sqrt(d_k) like forward)
    Tensor dS_raw = softmax_backward_2d(A, dA);  // (seq, seq)
    // Scale: dS = dS_raw / temperature (since S was divided by temperature in forward)
    for (int i = 0; i < dS_raw.total_size; ++i)
        dS_raw.data[i] /= temperature;

    // ── Backward through QK^T ─────────────────────────────────
    // dQ = dS @ K
    // dK = dS^T @ Q
    Tensor dQ  = vedic_gemm(dS_raw, K);          // (seq, d_k)
    Tensor dS_T = dS_raw.transpose();
    Tensor dK  = vedic_gemm(dS_T, Q);            // (seq, d_k)

    // ── dW_Q, dW_K, dW_V ─────────────────────────────────────
    Tensor X_T = X.transpose();
    Tensor dW_Q_c = vedic_gemm(X_T, dQ);
    for (int i = 0; i < dW_Q.total_size; ++i) dW_Q.data[i] += dW_Q_c.data[i];

    Tensor dW_K_c = vedic_gemm(X_T, dK);
    for (int i = 0; i < dW_K.total_size; ++i) dW_K.data[i] += dW_K_c.data[i];

    Tensor dW_V_c = vedic_gemm(X_T, dV);
    for (int i = 0; i < dW_V.total_size; ++i) dW_V.data[i] += dW_V_c.data[i];

    // ── dX from this head = dQ@W_Q^T + dK@W_K^T + dV@W_V^T ──
    Tensor W_Q_T = head.W_Q.transpose();
    Tensor W_K_T = head.W_K.transpose();
    Tensor W_V_T = head.W_V.transpose();

    Tensor dX_head = vedic_gemm(dQ, W_Q_T);
    Tensor dK_inp  = vedic_gemm(dK, W_K_T);
    Tensor dV_inp  = vedic_gemm(dV, W_V_T);

    for (int i = 0; i < dX_head.total_size; ++i)
        dX_head.data[i] += dK_inp.data[i] + dV_inp.data[i];

    return dX_head;  // (seq, d_model)
}

// ── 8. Full TransformerBlock backward ────────────────────────
// Forward:
//   normed1  = ln1(X)
//   attn_out = MHA(normed1)
//   h        = divergence_free(X + attn_out)   [residual 1]
//   normed2  = ln2(h)
//   ffn_out  = FFN(normed2)
//   Y        = divergence_free(h + ffn_out)     [residual 2]
//
// Backward (chain rule through residuals):
//   dh_total  = dY + dY_ffn_residual
//   ... etc.
//
// Note on divergence_free backward:
//   divergence_free(U) = U - mean(U)  per row.
//   Backward: dU = dY - mean(dY) per row  (same operation — self-adjoint).
struct BlockGrads {
    Tensor dW_Q, dW_K, dW_V, dW_O;   // per-head (summed across heads)
    Tensor dW_proj;
    Tensor dW1, db1, dW2, db2;
    Tensor dln1_gamma, dln1_beta;
    Tensor dln2_gamma, dln2_beta;
};

inline Tensor divergence_free_backward(const Tensor& dY) {
    // divergence_free: Y = X - mean(X, dim=cols)
    // Backward: dX = dY - mean(dY, dim=cols)
    int seq = dY.rows(), dim = dY.cols();
    Tensor dX(dY.shape, 0.0f);
    for (int i = 0; i < seq; ++i) {
        float mean_dY = 0.0f;
        for (int j = 0; j < dim; ++j) mean_dY += dY.at(i,j);
        mean_dY /= dim;
        for (int j = 0; j < dim; ++j)
            dX.at(i,j) = dY.at(i,j) - mean_dY;
    }
    return dX;
}

// block_backward returns dX (gradient w.r.t. block input X).
// grads struct must be pre-allocated to matching shapes before call.
inline Tensor block_backward(TransformerBlock& block,
                              const Tensor& X,        // (seq, d_model) — block input
                              const Tensor& dY,       // (seq, d_model) — upstream
                              BlockGrads&   grads,
                              bool          is_training = true)
{
    int seq   = X.rows();
    int d     = X.cols();

    // ── Recompute forward pass intermediates ──────────────────
    // (needed for grad computation; same operations as TransformerBlock::forward)
    Tensor normed1  = block.ln1.forward(X, is_training);
    Tensor attn_out = block.mha.forward(normed1);
    Tensor h_pre    = X + attn_out;                    // before divergence_free
    Tensor h        = divergence_free(h_pre);          // residual 1 output

    Tensor normed2  = block.ln2.forward(h, is_training);
    // Recompute FFN pre-activation for GELU backward
    Tensor pre_act  = vedic_gemm_bias(normed2, block.ffn.W1, block.ffn.b1);
    Tensor ffn_out  = block.ffn.forward(normed2, is_training);
    // Y = divergence_free(h + ffn_out)  — dY is given

    // ── Step 1: backward through residual 2 + divergence_free ─
    // Y = divergence_free(h + ffn_out)
    // dh_plus_ffn = divergence_free_backward(dY)
    Tensor dh_plus_ffn = divergence_free_backward(dY);  // (seq, d)

    // d(h) from residual: dh_res2 = dh_plus_ffn (pass-through)
    // d(ffn_out) = dh_plus_ffn (same — additive residual)
    Tensor dh_from_res2 = dh_plus_ffn;
    Tensor d_ffn_out    = dh_plus_ffn;

    // ── Step 2: backward through FFN ──────────────────────────
    // FFN: normed2 → pre_act → GELU → W2 → ffn_out
    Tensor d_normed2(normed2.shape, 0.0f);
    Tensor d_from_ffn = ffn_backward(block.ffn, normed2, pre_act, d_ffn_out,
                                      grads.dW1, grads.db1,
                                      grads.dW2, grads.db2);
    // d_normed2 += d_from_ffn
    for (int i = 0; i < d_normed2.total_size; ++i)
        d_normed2.data[i] += d_from_ffn.data[i];

    // ── Step 3: backward through ln2 ──────────────────────────
    Tensor dh_from_ln2(h.shape, 0.0f);
    layernorm_backward(h, d_normed2, block.ln2.gamma, block.ln2.eps,
                       dh_from_ln2, grads.dln2_gamma, grads.dln2_beta);

    // Total dh = from residual2 + from ln2
    Tensor dh(h.shape, 0.0f);
    for (int i = 0; i < dh.total_size; ++i)
        dh.data[i] = dh_from_res2.data[i] + dh_from_ln2.data[i];

    // ── Step 4: backward through residual 1 + divergence_free ─
    // h = divergence_free(X + attn_out)
    Tensor d_X_plus_attn = divergence_free_backward(dh);   // (seq, d)

    // dX from residual (pass-through): dX += d_X_plus_attn
    // d_attn_out = d_X_plus_attn
    Tensor dX_res1      = d_X_plus_attn;
    Tensor d_attn_out   = d_X_plus_attn;

    // ── Step 5: backward through MHA ──────────────────────────
    // MHA: normed1 → heads → concat → W_proj → attn_out
    //
    // Backward through W_proj first
    // attn_out = concat @ W_proj
    // d_concat = d_attn_out @ W_proj^T
    // dW_proj  += concat^T @ d_attn_out
    int num_heads = block.mha.num_heads;
    int d_k       = block.mha.d_k;

    // Recompute concat (head outputs concatenated)
    Tensor concat({seq, d}, 0.0f);
    for (int hh = 0; hh < num_heads; ++hh) {
        Tensor head_out = block.mha.heads[hh].forward(normed1);
        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < d_k; ++j)
                concat.at(i, hh * d_k + j) = head_out.at(i, j);
    }

    // dW_proj += concat^T @ d_attn_out
    Tensor concat_T    = concat.transpose();
    Tensor dW_proj_c   = vedic_gemm(concat_T, d_attn_out);
    for (int i = 0; i < grads.dW_proj.total_size; ++i)
        grads.dW_proj.data[i] += dW_proj_c.data[i];

    // d_concat = d_attn_out @ W_proj^T
    Tensor W_proj_T = block.mha.W_proj.transpose();
    Tensor d_concat = vedic_gemm(d_attn_out, W_proj_T);  // (seq, d)

    // ── Step 6: backward through each head ────────────────────
    // Each head takes normed1 as input, outputs d_k columns of concat.
    // Slice d_concat into per-head upstream grads.
    Tensor d_normed1(normed1.shape, 0.0f);
    for (int hh = 0; hh < num_heads; ++hh) {
        // Slice: d_concat[:, hh*d_k : (hh+1)*d_k]
        Tensor dY_head({seq, d_k}, 0.0f);
        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < d_k; ++j)
                dY_head.at(i,j) = d_concat.at(i, hh * d_k + j);

        // Per-head weight grads (accumulate into BlockGrads)
        // Note: grads.dW_Q etc. are (d_model, d_k) — shared across heads here
        // (they sum across heads — each head contributes its slice)
        Tensor dX_head = attention_head_backward(
            block.mha.heads[hh], normed1, dY_head,
            grads.dW_Q, grads.dW_K, grads.dW_V, grads.dW_O);

        for (int i = 0; i < d_normed1.total_size; ++i)
            d_normed1.data[i] += dX_head.data[i];
    }

    // ── Step 7: backward through ln1 ──────────────────────────
    Tensor dX_from_ln1(X.shape, 0.0f);
    layernorm_backward(X, d_normed1, block.ln1.gamma, block.ln1.eps,
                       dX_from_ln1, grads.dln1_gamma, grads.dln1_beta);

    // ── Total dX ──────────────────────────────────────────────
    Tensor dX(X.shape, 0.0f);
    for (int i = 0; i < dX.total_size; ++i)
        dX.data[i] = dX_res1.data[i] + dX_from_ln1.data[i];

    return dX;
}

// ── 9. Full model backward ────────────────────────────────────
// Ties everything together for the training loop.
// Fills grads_out (same order as model.parameters()).
//
// params order (matches LOGOSModel::parameters()):
//   [0] embedding   {vocab, d}
//   [1] lm_head     {d, vocab}
//   [2..2+L*N_per_layer-1] layer weights (per TransformerBlock::parameters())
//
// After this call: grads_out is fully populated with real analytical
// gradients for ALL parameters. Pass directly to clip_gradients() + optimizer.
//
// Returns: scalar cross-entropy loss (averaged over seq tokens).
inline float model_backward(LOGOSModel&              model,
                             const std::vector<int>&  input_ids,
                             const std::vector<int>&  target_ids,
                             std::vector<Tensor>&     grads_out,   // pre-allocated
                             bool                     is_training = true)
{
    int seq   = (int)input_ids.size();
    int vocab = model.cfg.vocab_size;
    int d     = model.cfg.d_model;
    int L     = model.cfg.num_layers;

    // ── Forward pass with intermediate storage ────────────────
    // Collect per-layer inputs for backward
    std::vector<Tensor> layer_inputs(L + 1);  // layer_inputs[0] = after RoPE

    // Embedding lookup
    Tensor X({seq, d}, 0.0f);
    for (int i = 0; i < seq; ++i) {
        int tok = std::max(0, std::min(input_ids[i], vocab-1));
        for (int j = 0; j < d; ++j) X.at(i,j) = model.embedding.at(tok,j);
    }
    apply_rope(X, seq, d);
    layer_inputs[0] = X;

    // Forward through layers, saving inputs
    for (int l = 0; l < L; ++l) {
        layer_inputs[l+1] = model.layers[l].forward(layer_inputs[l], is_training);
    }
    Tensor X_final = layer_inputs[L];

    // ── Loss + dLogits ────────────────────────────────────────
    Tensor logits = vedic_gemm(X_final, model.lm_head);
    float  loss   = 0.0f;
    Tensor dLogits = softmax_ce_backward(logits, target_ids, loss);

    if (std::isnan(loss) || std::isinf(loss)) return loss;

    // ── lm_head grad: grads_out[1] += X_final^T @ dLogits ────
    Tensor X_T     = X_final.transpose();
    Tensor dW_lm_c = vedic_gemm(X_T, dLogits);
    for (int i = 0; i < grads_out[1].total_size; ++i)
        grads_out[1].data[i] += dW_lm_c.data[i];

    // ── dX_final = dLogits @ lm_head^T ───────────────────────
    Tensor lm_T  = model.lm_head.transpose();
    Tensor dX    = vedic_gemm(dLogits, lm_T);    // (seq, d)

    // ── Backward through transformer layers ───────────────────
    // Param index tracking (matches LOGOSModel::parameters() order):
    // [0]=embedding, [1]=lm_head, then per-block in block.parameters() order.
    // TransformerBlock::parameters() order:
    //   MHA: W_proj, then per-head (W_Q, W_K, W_V, W_O)
    //   FFN: W1, b1, W2, b2
    //   ln1: gamma, beta
    //   ln2: gamma, beta
    // Total per block = 1 + 4*num_heads + 4 + 4 = 9 + 4*H tensors

    int num_heads = model.cfg.num_heads;
    int tensors_per_block = 1 + 4*num_heads + 4 + 4;  // W_proj + heads + FFN + LN×2
    int param_base = 2;  // after embedding and lm_head

    for (int l = L - 1; l >= 0; --l) {
        TransformerBlock& blk = model.layers[l];
        int block_offset = param_base + l * tensors_per_block;

        // Allocate BlockGrads pointing into grads_out slices
        // We build grad views manually (matching block.parameters() order)
        // Index layout within block:
        //   [0]        = W_proj  {d, d}
        //   [1..4H]    = per-head W_Q, W_K, W_V, W_O (each {d, d_k} or {d_k,d})
        //   [4H+1..+4] = W1, b1, W2, b2
        //   [4H+5..+8] = ln1.gamma, ln1.beta, ln2.gamma, ln2.beta
        BlockGrads bg;
        int dk = model.cfg.d_model / num_heads;

        // W_proj
        bg.dW_proj     = Tensor({d, d}, 0.0f);
        // Per-head: aggregate all heads' grads into one tensor per param type
        bg.dW_Q        = Tensor({d, dk}, 0.0f);
        bg.dW_K        = Tensor({d, dk}, 0.0f);
        bg.dW_V        = Tensor({d, dk}, 0.0f);
        bg.dW_O        = Tensor({dk, d}, 0.0f);
        // FFN
        int d_ff = blk.ffn.d_ff;
        bg.dW1         = Tensor({d, d_ff}, 0.0f);
        bg.db1         = Tensor({1, d_ff}, 0.0f);
        bg.dW2         = Tensor({d_ff, d}, 0.0f);
        bg.db2         = Tensor({1, d}, 0.0f);
        // LN
        bg.dln1_gamma  = Tensor({1, d}, 0.0f);
        bg.dln1_beta   = Tensor({1, d}, 0.0f);
        bg.dln2_gamma  = Tensor({1, d}, 0.0f);
        bg.dln2_beta   = Tensor({1, d}, 0.0f);

        // Block backward
        dX = block_backward(blk, layer_inputs[l], dX, bg, is_training);

        // Accumulate into grads_out at correct offsets
        // block.parameters() order: W_proj, then heads[h].W_Q/K/V/O, FFN, LN
        auto acc = [](Tensor& dst, const Tensor& src) {
            for (int i = 0; i < dst.total_size && i < src.total_size; ++i)
                dst.data[i] += src.data[i];
        };

        acc(grads_out[block_offset], bg.dW_proj);
        for (int hh = 0; hh < num_heads; ++hh) {
            int ho = block_offset + 1 + hh * 4;
            acc(grads_out[ho+0], bg.dW_Q);
            acc(grads_out[ho+1], bg.dW_K);
            acc(grads_out[ho+2], bg.dW_V);
            acc(grads_out[ho+3], bg.dW_O);
        }
        int fbase = block_offset + 1 + num_heads * 4;
        acc(grads_out[fbase+0], bg.dW1);
        acc(grads_out[fbase+1], bg.db1);
        acc(grads_out[fbase+2], bg.dW2);
        acc(grads_out[fbase+3], bg.db2);
        int lbase = fbase + 4;
        acc(grads_out[lbase+0], bg.dln1_gamma);
        acc(grads_out[lbase+1], bg.dln1_beta);
        acc(grads_out[lbase+2], bg.dln2_gamma);
        acc(grads_out[lbase+3], bg.dln2_beta);
    }

    // ── Embedding grad ────────────────────────────────────────
    // dX is now gradient w.r.t. RoPE output = gradient w.r.t. embedding output
    // (RoPE is parameter-free — gradient passes through unchanged in magnitude)
    for (int si = 0; si < seq; ++si) {
        int tok = std::max(0, std::min(input_ids[si], vocab-1));
        for (int di = 0; di < d; ++di)
            grads_out[0].at(tok, di) += dX.at(si, di);
    }

    return loss;
}
