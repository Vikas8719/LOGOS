#pragma once

#include "Tensor.hpp"
#include "VedicGEMM.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <unordered_map>


inline Tensor boltzmann_softmax(const Tensor& scores, float temperature) {
    if (temperature <= 0.0f)
        throw std::invalid_argument("boltzmann_softmax: temperature must be > 0");
    int seq = scores.rows(), len = scores.cols();
    Tensor probs(scores.shape);
    for (int i = 0; i < seq; ++i) {
        float max_val = scores.at(i, 0);
        for (int j = 1; j < len; ++j)
            max_val = std::max(max_val, scores.at(i, j));
        float sum = 0.0f;
        for (int j = 0; j < len; ++j) {
            float e = std::exp((scores.at(i, j) - max_val) / temperature);
            probs.at(i, j) = e;
            sum += e;
        }
        float safe_sum = std::max(sum, 1e-9f);
        for (int j = 0; j < len; ++j)
            probs.at(i, j) /= safe_sum;
    }
    return probs;
}


inline void clear_mask_caches() {
    static auto& causal = []() -> std::unordered_map<int, Tensor>& {
        static std::unordered_map<int, Tensor> c; return c;
    }();
    static auto& sparse = []() -> std::unordered_map<int64_t, Tensor>& {
        static std::unordered_map<int64_t, Tensor> s; return s;
    }();
    causal.clear();
    sparse.clear();
}

inline const Tensor& causal_mask_cached(int seq_len) {
    static std::unordered_map<int, Tensor> mask_cache;
    auto it = mask_cache.find(seq_len);
    if (it != mask_cache.end()) return it->second;

    Tensor mask({seq_len, seq_len}, 0.0f);
    for (int i = 0; i < seq_len; ++i)
        for (int j = i + 1; j < seq_len; ++j)
            mask.at(i, j) = -1e9f;

    mask_cache.emplace(seq_len, std::move(mask));
    return mask_cache.at(seq_len);
}


inline Tensor causal_mask(int seq_len) {
    return causal_mask_cached(seq_len);  // delegate to cache
}


inline Tensor shunyam_sparse_mask(int seq_len,
                                   int window = 8,
                                   int stride = 4)
{
    Tensor mask({seq_len, seq_len}, -1e9f);  

    for (int i = 0; i < seq_len; ++i) {
        
        int local_start = std::max(0, i - window);
        for (int j = local_start; j <= i; ++j)
            mask.at(i, j) = 0.0f;  

        
        for (int j = 0; j < i; j += stride)
            mask.at(i, j) = 0.0f;  
    }
    return mask;
}


inline const Tensor& shunyam_sparse_mask_cached(int seq_len,
                                                  int window = 8,
                                                  int stride = 4)
{
    static std::unordered_map<int64_t, Tensor> sparse_cache;
    int64_t key = (int64_t)seq_len * 1000000LL + (int64_t)window * 1000LL + (int64_t)stride;
    auto it = sparse_cache.find(key);
    if (it != sparse_cache.end()) return it->second;
    sparse_cache.emplace(key, shunyam_sparse_mask(seq_len, window, stride));
    return sparse_cache.at(key);
}



inline Tensor navier_stokes_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                       const Tensor& mask,
                                       float temperature,
                                       float eta  = 0.1f,    // advection
                                       float nu   = 0.05f)   // viscosity
{
    int seq = Q.rows(), d_k = Q.cols(), d_v = V.cols();

   
    Tensor Q_adv(Q.shape, 0.0f);
    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < d_k; ++j) {
            float q_i   = Q.at(i, j);
            float q_im1 = (i > 0) ? Q.at(i-1, j) : q_i;  // boundary: no flow at i=0
            float q_vel = q_i - q_im1;
            Q_adv.at(i, j) = q_i + eta * q_vel;
        }
    }

   
    Tensor K_T    = K.transpose();
    Tensor scores = vedic_gemm(Q_adv, K_T);
    scores += mask;  // causal / sparse mask

    Tensor attn_weights = boltzmann_softmax(scores, temperature);

   
    Tensor V_smooth(V.shape, 0.0f);
    for (int i = 0; i < seq; ++i) {
        int i_left  = (i > 0)      ? i - 1 : 0;
        int i_right = (i < seq-1)  ? i + 1 : seq - 1;
        for (int j = 0; j < d_v; ++j) {
            V_smooth.at(i, j) = (1.0f - nu) * V.at(i, j)
                               + (nu * 0.5f) * (V.at(i_left, j) + V.at(i_right, j));
        }
    }

   
    return vedic_gemm(attn_weights, V_smooth);
}

struct AttentionHead {
    Tensor W_Q, W_K, W_V, W_O;
    int d_model, d_k;

    AttentionHead(int d_model_, int d_k_)
        : d_model(d_model_), d_k(d_k_),
          W_Q({d_model_, d_k_}), W_K({d_model_, d_k_}),
          W_V({d_model_, d_k_}), W_O({d_k_, d_model_})
    {
        float scale = std::sqrt(2.0f / d_model_);
        W_Q.fill_random(-scale, scale);
        W_K.fill_random(-scale, scale);
        W_V.fill_random(-scale, scale);
        W_O.fill_random(-scale, scale);
    }

    bool  use_sparse     = true;
    int   sparse_window  = 8;
    int   sparse_stride  = 4;
   
    bool  use_navier_stokes = false;
    float ns_alpha          = 0.3f;   
    float ns_eta            = 0.1f;   
    float ns_nu             = 0.05f;  

    Tensor forward(const Tensor& X) {
        float temperature = std::sqrt(static_cast<float>(d_k));

        Tensor Q = vedic_gemm(X, W_Q);
        Tensor K = vedic_gemm(X, W_K);
        Tensor V = vedic_gemm(X, W_V);

        int seq = X.rows();
        const Tensor* mask_ptr = nullptr;
        if (use_sparse && seq > sparse_window) {
            mask_ptr = &shunyam_sparse_mask_cached(seq, sparse_window, sparse_stride);
        } else {
            mask_ptr = &causal_mask_cached(seq);
        }

        if (use_navier_stokes && ns_alpha > 0.0f) {
            
            Tensor K_T       = K.transpose();
            Tensor scores_std = vedic_gemm(Q, K_T);
            scores_std       += *mask_ptr;
            Tensor w_std      = boltzmann_softmax(scores_std, temperature);
            Tensor out_std    = vedic_gemm(w_std, V);
            Tensor out_std_O  = vedic_gemm(out_std, W_O);

            
            Tensor out_ns = navier_stokes_attention(Q, K, V, *mask_ptr,
                                                     temperature, ns_eta, ns_nu);
            Tensor out_ns_O = vedic_gemm(out_ns, W_O);

            
            Tensor blended({seq, d_model}, 0.0f);
            float w1 = 1.0f - ns_alpha, w2 = ns_alpha;
            for (int i = 0; i < blended.total_size; ++i)
                blended.data[i] = w1 * out_std_O.data[i] + w2 * out_ns_O.data[i];
            return blended;
        } else {
            
            Tensor K_T    = K.transpose();
            Tensor scores = vedic_gemm(Q, K_T);
            scores       += *mask_ptr;
            Tensor attn_weights = boltzmann_softmax(scores, temperature);
            Tensor output       = vedic_gemm(attn_weights, V);
            return vedic_gemm(output, W_O);
        }
    }

    std::vector<Tensor*> parameters() { return {&W_Q, &W_K, &W_V, &W_O}; }
};


struct MultiHeadAttention {
    std::vector<AttentionHead> heads;
    Tensor W_proj;
    int num_heads, d_model, d_k;

    MultiHeadAttention(int d_model_, int num_heads_)
        : num_heads(num_heads_), d_model(d_model_),
          d_k(d_model_ / num_heads_),
          W_proj({d_model_, d_model_})
    {
        if (d_model_ % num_heads_ != 0)
            throw std::invalid_argument("d_model must be divisible by num_heads");
        for (int h = 0; h < num_heads_; ++h)
            heads.emplace_back(d_model_, d_k);
        float scale = std::sqrt(2.0f / d_model_);
        W_proj.fill_random(-scale, scale);
    }

    Tensor forward(const Tensor& X) {
        int seq = X.rows();
        Tensor concat({seq, d_model}, 0.0f);
        for (int h = 0; h < num_heads; ++h) {
            Tensor head_out = heads[h].forward(X);
            for (int i = 0; i < seq; ++i)
                for (int j = 0; j < d_k; ++j)
                    concat.at(i, h * d_k + j) = head_out.at(i, j);
        }
        return vedic_gemm(concat, W_proj);
    }

    std::vector<Tensor*> parameters() {
        std::vector<Tensor*> params = {&W_proj};
        for (auto& h : heads)
            for (auto* p : h.parameters())
                params.push_back(p);
        return params;
    }
};
