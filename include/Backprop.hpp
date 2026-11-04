#pragma once

#include "Tensor.hpp"
#include "VedicGEMM.hpp"
#include "Attention.hpp"
#include "FeedForward.hpp"
#include "LayerNorm.hpp"
#include "TransformerBlock.hpp"
#include "Model.hpp"
#include <cmath>
#include <vector>


inline void gemm_backward(const Tensor& A,   
                           const Tensor& B,   
                           const Tensor& dC,  
                           Tensor& dA,        
                           Tensor& dB)        
{
   
    Tensor B_T = B.transpose();   
    Tensor dA_contrib = vedic_gemm(dC, B_T);  
        for (int i = 0; i < dA.total_size; ++i) dA.data[i] += dA_contrib.data[i];

    Tensor A_T = A.transpose();   // (K, M)
    Tensor dB_contrib = vedic_gemm(A_T, dC);  // (K, N)
    for (int i = 0; i < dB.total_size; ++i) dB.data[i] += dB_contrib.data[i];
}


inline Tensor gelu_backward(const Tensor& X,  
                              const Tensor& dY) 
{
    constexpr float c  = 0.7978845608f;
    constexpr float a  = 0.044715f;
    Tensor dX(X.shape, 0.0f);
    for (int i = 0; i < X.total_size; ++i) {
        float x  = X.data[i];
        float x3 = x * x * x;
        float t  = std::tanh(c * (x + a * x3));
        float dt = 1.0f - t * t;  // sech^2
        float dgelu = 0.5f * (1.0f + t)
                    + 0.5f * x * dt * (c * (1.0f + 3.0f * a * x * x));
        dX.data[i] = dY.data[i] * dgelu;
    }
    return dX;
}


inline Tensor bias_backward(const Tensor& dY)  // (seq, dim)
{
    int seq = dY.rows(), dim = dY.cols();
    Tensor dbias({1, dim}, 0.0f);
    for (int i = 0; i < seq; ++i)
        for (int j = 0; j < dim; ++j)
            dbias.at(0, j) += dY.at(i, j);
    return dbias;
}


inline void layernorm_backward(const Tensor&  X,      
                                const Tensor&  dY,     
                                const Tensor&  gamma,  
                                float          eps,
                                Tensor& dX,            
                                Tensor& dgamma,        
                                Tensor& dbeta)         
{
    int seq = X.rows(), D = X.cols();
    for (int i = 0; i < seq; ++i) {
        
        float mean = 0.0f, M2 = 0.0f;
        for (int j = 0; j < D; ++j) {
            float delta = X.at(i,j) - mean;
            mean += delta / (j + 1);
            M2   += delta * (X.at(i,j) - mean);
        }
        float var   = M2 / D;
        float std_  = std::sqrt(var + eps);
        float inv_s = 1.0f / std_;

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

inline Tensor ffn_backward(const FeedForward& ffn,
                            const Tensor& X,        // (seq, d_model)  — FFN input
                            const Tensor& pre_act,  // (seq, d_ff)     — pre-GELU
                            const Tensor& dY,       // (seq, d_model)  — upstream
                            Tensor& dW1, Tensor& db1,
                            Tensor& dW2, Tensor& db2)
{
    Tensor W2_T   = ffn.W2.transpose();
    Tensor dH     = vedic_gemm(dY, W2_T);      // (seq, d_ff)

    Tensor H(pre_act.shape);
    for (int i = 0; i < pre_act.total_size; ++i)
        H.data[i] = gelu(pre_act.data[i]);
    Tensor H_T = H.transpose();
    Tensor dW2_c = vedic_gemm(H_T, dY);
    for (int i = 0; i < dW2.total_size; ++i) dW2.data[i] += dW2_c.data[i];

    Tensor db2_c = bias_backward(dY);
    for (int i = 0; i < db2.total_size; ++i) db2.data[i] += db2_c.data[i];

    Tensor dPre = gelu_backward(pre_act, dH);   // (seq, d_ff)

    Tensor X_T  = X.transpose();
    Tensor dW1_c = vedic_gemm(X_T, dPre);
    for (int i = 0; i < dW1.total_size; ++i) dW1.data[i] += dW1_c.data[i];

    Tensor db1_c = bias_backward(dPre);
    for (int i = 0; i < db1.total_size; ++i) db1.data[i] += db1_c.data[i];

    Tensor W1_T = ffn.W1.transpose();
    return vedic_gemm(dPre, W1_T);    // (seq, d_model)
}

inline Tensor softmax_backward_2d(const Tensor& A,   // (seq, seq) softmax output
                                   const Tensor& dA)  // (seq, seq) upstream
{
    int rows = A.rows(), cols = A.cols();
    Tensor dS(A.shape, 0.0f);
    for (int i = 0; i < rows; ++i) {
        float dot = 0.0f;
        for (int k = 0; k < cols; ++k) dot += dA.at(i,k) * A.at(i,k);
        for (int j = 0; j < cols; ++j)
            dS.at(i,j) = A.at(i,j) * (dA.at(i,j) - dot);
    }
    return dS;
}

// [v40-BUG3-FIX] AttentionHead mein saved Q/K/V/A cache store karo
// Ye forward pass ke baad fill hoga — backward mein recompute nahi hoga
struct AttnFwdCache {
    Tensor Q, K, V, A;   // (seq, d_k) each; A is (seq, seq)
    bool valid = false;
};

// [v40-BUG3-FIX] Cache-aware backward — agar cache valid hai toh recompute nahi
// saved_cache: forward pass mein bhara hua Q/K/V/A
// Agar saved_cache.valid == false → fallback to full recompute (old behavior)
inline Tensor attention_head_backward(
    AttentionHead& head,
    const Tensor& X,            // (seq, d_model) — block input to this head
    const Tensor& dO,           // (seq, d_k)     — upstream grad w.r.t. head output O
    Tensor& dW_Q, Tensor& dW_K, Tensor& dW_V, Tensor& dW_O,
    const AttnFwdCache* saved_cache = nullptr)  // [v40-BUG3] optional cache
{
    float temperature = std::sqrt((float)head.d_k);
    int   seq         = X.rows();

    // [v40-BUG3-FIX] Cache hit → recompute avoid karo (Flash-style)
    // Cache miss → old fallback (correctness preserve)
    Tensor Q, K, V, A;
    if (saved_cache && saved_cache->valid) {
        // Saved Q/K/V/A use karo — seq² memory + recompute dono bachte hain
        Q = saved_cache->Q;
        K = saved_cache->K;
        V = saved_cache->V;
        A = saved_cache->A;
    } else {
        // Fallback: full recompute (pehle wala behavior)
        Q = vedic_gemm(X, head.W_Q);  // (seq, d_k)
        K = vedic_gemm(X, head.W_K);  // (seq, d_k)
        V = vedic_gemm(X, head.W_V);  // (seq, d_k)

        Tensor mask;
        if (head.use_sparse && seq > head.sparse_window)
            mask = shunyam_sparse_mask_cached(seq, head.sparse_window, head.sparse_stride);
        else
            mask = causal_mask_cached(seq);

        Tensor K_T = K.transpose();
        Tensor S   = vedic_gemm(Q, K_T);
        S += mask;
        A = boltzmann_softmax(S, temperature);  // (seq, seq)
    }

    Tensor V_T = V.transpose();                       // (d_k, seq)
    Tensor dA  = vedic_gemm(dO, V_T);                // (seq, seq) ✅

    Tensor A_T = A.transpose();                       // (seq, seq)
    Tensor dV  = vedic_gemm(A_T, dO);                // (seq, d_k) ✅

    Tensor dS_raw = softmax_backward_2d(A, dA);      // (seq, seq)
    for (int i = 0; i < dS_raw.total_size; ++i)
        dS_raw.data[i] /= temperature;

    Tensor dQ   = vedic_gemm(dS_raw, K);             // (seq, d_k)
    Tensor dS_T = dS_raw.transpose();
    Tensor dK   = vedic_gemm(dS_T, Q);               // (seq, d_k)

    Tensor X_T    = X.transpose();                    // (d_model, seq)
    Tensor dW_Q_c = vedic_gemm(X_T, dQ);             // (d_model, d_k) ✅
    Tensor dW_K_c = vedic_gemm(X_T, dK);             // (d_model, d_k) ✅
    Tensor dW_V_c = vedic_gemm(X_T, dV);             // (d_model, d_k) ✅

    for (int i = 0; i < dW_Q.total_size; ++i) dW_Q.data[i] += dW_Q_c.data[i];
    for (int i = 0; i < dW_K.total_size; ++i) dW_K.data[i] += dW_K_c.data[i];
    for (int i = 0; i < dW_V.total_size; ++i) dW_V.data[i] += dW_V_c.data[i];

    (void)dW_O;

    Tensor W_Q_T   = head.W_Q.transpose();            // (d_k, d_model)
    Tensor W_K_T   = head.W_K.transpose();
    Tensor W_V_T   = head.W_V.transpose();

    Tensor dX_head = vedic_gemm(dQ, W_Q_T);          // (seq, d_model) ✅
    Tensor dK_inp  = vedic_gemm(dK, W_K_T);          // (seq, d_model) ✅
    Tensor dV_inp  = vedic_gemm(dV, W_V_T);          // (seq, d_model) ✅

    for (int i = 0; i < dX_head.total_size; ++i)
        dX_head.data[i] += dK_inp.data[i] + dV_inp.data[i];

    return dX_head;  // (seq, d_model) ✅
}

struct BlockGrads {
    Tensor dW_Q, dW_K, dW_V, dW_O;   // per-head (summed across heads)
    Tensor dW_proj;
    Tensor dW1, db1, dW2, db2;
    Tensor dln1_gamma, dln1_beta;
    Tensor dln2_gamma, dln2_beta;
};

inline Tensor divergence_free_backward(const Tensor& dY) {
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

inline Tensor block_backward(TransformerBlock& block,
                              const Tensor& X,        // (seq, d_model) — block input
                              const Tensor& dY,       // (seq, d_model) — upstream
                              BlockGrads&   grads,
                              bool          is_training = true)
{
    int seq   = X.rows();
    int d     = X.cols();

    Tensor normed1  = block.ln1.forward(X, is_training);
    Tensor attn_out = block.mha.forward(normed1);
    Tensor h_pre    = X + attn_out;                    // before divergence_free
    Tensor h        = divergence_free(h_pre);          // residual 1 output

    Tensor normed2  = block.ln2.forward(h, is_training);
    Tensor pre_act  = vedic_gemm_bias(normed2, block.ffn.W1, block.ffn.b1);
    Tensor ffn_out  = block.ffn.forward(normed2, is_training);

    Tensor dh_plus_ffn = divergence_free_backward(dY);  // (seq, d)

    Tensor dh_from_res2 = dh_plus_ffn;
    Tensor d_ffn_out    = dh_plus_ffn;

    Tensor d_normed2(normed2.shape, 0.0f);
    Tensor d_from_ffn = ffn_backward(block.ffn, normed2, pre_act, d_ffn_out,
                                      grads.dW1, grads.db1,
                                      grads.dW2, grads.db2);
    for (int i = 0; i < d_normed2.total_size; ++i)
        d_normed2.data[i] += d_from_ffn.data[i];

    Tensor dh_from_ln2(h.shape, 0.0f);
    layernorm_backward(h, d_normed2, block.ln2.gamma, block.ln2.eps,
                       dh_from_ln2, grads.dln2_gamma, grads.dln2_beta);

    Tensor dh(h.shape, 0.0f);
    for (int i = 0; i < dh.total_size; ++i)
        dh.data[i] = dh_from_res2.data[i] + dh_from_ln2.data[i];

    Tensor d_X_plus_attn = divergence_free_backward(dh);   // (seq, d)

    Tensor dX_res1      = d_X_plus_attn;
    Tensor d_attn_out   = d_X_plus_attn;

    int num_heads = block.mha.num_heads;
    int d_k       = block.mha.d_k;

    // [v40-BUG3-FIX] cached head outputs use karo — agar cache valid hai
    // Block forward mein forward_cached call hona chahiye
    Tensor concat({seq, d}, 0.0f);
    for (int hh = 0; hh < num_heads; ++hh) {
        // forward_cached se saved output use karo (cache valid hai toh)
        // Agar cache nahi hai toh fallback forward call
        Tensor head_out;
        if (block.mha.heads[hh].fwd_cache.valid) {
            // A*V already computed — cache se reconstruct karo
            head_out = vedic_gemm(block.mha.heads[hh].fwd_cache.A,
                                   block.mha.heads[hh].fwd_cache.V);
        } else {
            head_out = block.mha.heads[hh].forward(normed1);
        }
        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < d_k; ++j)
                concat.at(i, hh * d_k + j) = head_out.at(i, j);
    }

    Tensor concat_T    = concat.transpose();
    Tensor dW_proj_c   = vedic_gemm(concat_T, d_attn_out);
    for (int i = 0; i < grads.dW_proj.total_size; ++i)
        grads.dW_proj.data[i] += dW_proj_c.data[i];

    Tensor W_proj_T = block.mha.W_proj.transpose();
    Tensor d_concat = vedic_gemm(d_attn_out, W_proj_T);  // (seq, d)

    Tensor d_normed1(normed1.shape, 0.0f);
    for (int hh = 0; hh < num_heads; ++hh) {
        Tensor dY_head({seq, d_k}, 0.0f);
        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < d_k; ++j)
                dY_head.at(i,j) = d_concat.at(i, hh * d_k + j);

        // [v40-BUG3-FIX] Cache-aware backward call — saved Q/K/V/A pass karo
        const AttnFwdCache* cache_ptr = nullptr;
        AttnFwdCache tmp_cache;
        if (block.mha.heads[hh].fwd_cache.valid) {
            tmp_cache.Q     = block.mha.heads[hh].fwd_cache.Q;
            tmp_cache.K     = block.mha.heads[hh].fwd_cache.K;
            tmp_cache.V     = block.mha.heads[hh].fwd_cache.V;
            tmp_cache.A     = block.mha.heads[hh].fwd_cache.A;
            tmp_cache.valid = true;
            cache_ptr = &tmp_cache;
        }

        Tensor dX_head = attention_head_backward(
            block.mha.heads[hh], normed1, dY_head,
            grads.dW_Q, grads.dW_K, grads.dW_V, grads.dW_O,
            cache_ptr);  // [v40-BUG3] cache pass karo

        for (int i = 0; i < d_normed1.total_size; ++i)
            d_normed1.data[i] += dX_head.data[i];
    }

    Tensor dX_from_ln1(X.shape, 0.0f);
    layernorm_backward(X, d_normed1, block.ln1.gamma, block.ln1.eps,
                       dX_from_ln1, grads.dln1_gamma, grads.dln1_beta);

    Tensor dX(X.shape, 0.0f);
    for (int i = 0; i < dX.total_size; ++i)
        dX.data[i] = dX_res1.data[i] + dX_from_ln1.data[i];

    return dX;
}

inline float model_backward_with_logits(LOGOSModel&              model,
                                        const std::vector<int>&  input_ids,
                                        const std::vector<int>&  target_ids,
                                        std::vector<Tensor>&     grads_out,
                                        Tensor&                  logits_out,
                                        bool                     is_training = true);

inline float model_backward(LOGOSModel&              model,
                             const std::vector<int>&  input_ids,
                             const std::vector<int>&  target_ids,
                             std::vector<Tensor>&     grads_out,
                             bool                     is_training = true)
{
    int seq   = (int)input_ids.size();
    int vocab = model.cfg.vocab_size;
    int d     = model.cfg.d_model;
    int L     = model.cfg.num_layers;

    std::vector<Tensor> layer_inputs(L + 1);  // layer_inputs[0] = after RoPE

    Tensor X({seq, d}, 0.0f);
    for (int i = 0; i < seq; ++i) {
        int tok = std::max(0, std::min(input_ids[i], vocab-1));
        for (int j = 0; j < d; ++j) X.at(i,j) = model.embedding.at(tok,j);
    }
    apply_rope(X, seq, d);
    layer_inputs[0] = X;

    // [v40-BUG3-FIX] forward_cached use karo — backward mein Q/K/V/A recompute nahi hoga
    for (int l = 0; l < L; ++l) {
        layer_inputs[l+1] = model.layers[l].forward_cached(layer_inputs[l], is_training);
    }
    Tensor X_final = layer_inputs[L];

    Tensor logits = vedic_gemm(X_final, model.lm_head);
    float  loss   = 0.0f;
    Tensor dLogits = softmax_ce_backward(logits, target_ids, loss);

    if (std::isnan(loss) || std::isinf(loss)) {
        // Cache clear karo on failure bhi
        for (int l = 0; l < L; ++l) model.layers[l].clear_attn_cache();
        return loss;
    }

    Tensor X_T     = X_final.transpose();
    Tensor dW_lm_c = vedic_gemm(X_T, dLogits);
    for (int i = 0; i < grads_out[1].total_size; ++i)
        grads_out[1].data[i] += dW_lm_c.data[i];

    Tensor lm_T  = model.lm_head.transpose();
    Tensor dX    = vedic_gemm(dLogits, lm_T);    // (seq, d)


    int num_heads = model.cfg.num_heads;
    int tensors_per_block = 1 + 4*num_heads + 4 + 4;  // W_proj + heads + FFN + LN×2
    int param_base = 2;  // after embedding and lm_head

    for (int l = L - 1; l >= 0; --l) {
        TransformerBlock& blk = model.layers[l];
        int block_offset = param_base + l * tensors_per_block;

        BlockGrads bg;
        int dk = model.cfg.d_model / num_heads;

        bg.dW_proj     = Tensor({d, d}, 0.0f);
        bg.dW_Q        = Tensor({d, dk}, 0.0f);
        bg.dW_K        = Tensor({d, dk}, 0.0f);
        bg.dW_V        = Tensor({d, dk}, 0.0f);
        bg.dW_O        = Tensor({dk, d}, 0.0f);
        int d_ff = blk.ffn.d_ff;
        bg.dW1         = Tensor({d, d_ff}, 0.0f);
        bg.db1         = Tensor({1, d_ff}, 0.0f);
        bg.dW2         = Tensor({d_ff, d}, 0.0f);
        bg.db2         = Tensor({1, d}, 0.0f);
        bg.dln1_gamma  = Tensor({1, d}, 0.0f);
        bg.dln1_beta   = Tensor({1, d}, 0.0f);
        bg.dln2_gamma  = Tensor({1, d}, 0.0f);
        bg.dln2_beta   = Tensor({1, d}, 0.0f);

        dX = block_backward(blk, layer_inputs[l], dX, bg, is_training);

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

    for (int si = 0; si < seq; ++si) {
        int tok = std::max(0, std::min(input_ids[si], vocab-1));
        for (int di = 0; di < d; ++di)
            grads_out[0].at(tok, di) += dX.at(si, di);
    }

    // [v40-BUG3-FIX] Cache clear karo — next step ke liye fresh start
    for (int l = 0; l < L; ++l) model.layers[l].clear_attn_cache();

    return loss;
}

inline float model_backward_with_logits(LOGOSModel&              model,
                                        const std::vector<int>&  input_ids,
                                        const std::vector<int>&  target_ids,
                                        std::vector<Tensor>&     grads_out,
                                        Tensor&                  logits_out,
                                        bool                     is_training)
{
    int seq   = (int)input_ids.size();
    int vocab = model.cfg.vocab_size;
    int d     = model.cfg.d_model;
    int L     = model.cfg.num_layers;

    std::vector<Tensor> layer_inputs(L + 1);

    Tensor X({seq, d}, 0.0f);
    for (int i = 0; i < seq; ++i) {
        int tok = std::max(0, std::min(input_ids[i], vocab-1));
        for (int j = 0; j < d; ++j) X.at(i,j) = model.embedding.at(tok,j);
    }
    apply_rope(X, seq, d);
    layer_inputs[0] = X;

    // [v40-BUG3-FIX] forward_cached use karo — attn Q/K/V/A cache fill hoga
    for (int l = 0; l < L; ++l)
        layer_inputs[l+1] = model.layers[l].forward_cached(layer_inputs[l], is_training);

    Tensor X_final = layer_inputs[L];
    logits_out = vedic_gemm(X_final, model.lm_head);

    float  loss    = 0.0f;
    Tensor dLogits = softmax_ce_backward(logits_out, target_ids, loss);

    if (std::isnan(loss) || std::isinf(loss)) {
        for (int l = 0; l < L; ++l) model.layers[l].clear_attn_cache();
        return loss;
    }

    Tensor X_T     = X_final.transpose();
    Tensor dW_lm_c = vedic_gemm(X_T, dLogits);
    for (int i = 0; i < grads_out[1].total_size; ++i)
        grads_out[1].data[i] += dW_lm_c.data[i];

    Tensor lm_T = model.lm_head.transpose();
    Tensor dX   = vedic_gemm(dLogits, lm_T);

    int num_heads       = model.cfg.num_heads;
    int tensors_per_block = 1 + 4*num_heads + 4 + 4;
    int param_base      = 2;

    for (int l = L - 1; l >= 0; --l) {
        TransformerBlock& blk = model.layers[l];
        int block_offset = param_base + l * tensors_per_block;

        BlockGrads bg;
        int dk   = d / num_heads;
        int d_ff = blk.ffn.d_ff;

        bg.dW_proj    = Tensor({d, d},     0.0f);
        bg.dW_Q       = Tensor({d, dk},    0.0f);
        bg.dW_K       = Tensor({d, dk},    0.0f);
        bg.dW_V       = Tensor({d, dk},    0.0f);
        bg.dW_O       = Tensor({dk, d},    0.0f);
        bg.dW1        = Tensor({d, d_ff},  0.0f);
        bg.db1        = Tensor({1, d_ff},  0.0f);
        bg.dW2        = Tensor({d_ff, d},  0.0f);
        bg.db2        = Tensor({1, d},     0.0f);
        bg.dln1_gamma = Tensor({1, d},     0.0f);
        bg.dln1_beta  = Tensor({1, d},     0.0f);
        bg.dln2_gamma = Tensor({1, d},     0.0f);
        bg.dln2_beta  = Tensor({1, d},     0.0f);

        dX = block_backward(blk, layer_inputs[l], dX, bg, is_training);

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

    for (int si = 0; si < seq; ++si) {
        int tok = std::max(0, std::min(input_ids[si], vocab-1));
        for (int di = 0; di < d; ++di)
            grads_out[0].at(tok, di) += dX.at(si, di);
    }

    // [v40-BUG3-FIX] Cache clear karo — next step ke liye
    for (int l = 0; l < L; ++l) model.layers[l].clear_attn_cache();

    return loss;
}
