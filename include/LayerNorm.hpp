#pragma once
#include "Tensor.hpp"
#include <cmath>
#include <vector>

struct LayerNorm {
    Tensor gamma, beta;
    int d_model;
    float eps = 1e-5f;

    LayerNorm(int d_model_) : d_model(d_model_),
        gamma({1, d_model_}, 1.0f),   // Init to 1 (scale)
        beta ({1, d_model_}, 0.0f)    // Init to 0 (shift)
    {}

    Tensor forward(const Tensor& X) const {
        int seq = X.rows(), dim = X.cols();
        if (seq == 0 || dim == 0) return Tensor(X.shape, 0.0f);
        Tensor Y(X.shape);

        for (int i = 0; i < seq; ++i) {
            float mean = 0.0f, M2 = 0.0f;
            for (int j = 0; j < dim; ++j) {
                float delta = X.at(i, j) - mean;
                mean += delta / (j + 1);              // running mean
                M2   += delta * (X.at(i, j) - mean); // running sum of sq deviations
            }
            float var     = M2 / dim;
            float inv_std = 1.0f / std::sqrt(var + eps);

            for (int j = 0; j < dim; ++j)
                Y.at(i, j) = gamma[j] * (X.at(i, j) - mean) * inv_std + beta[j];
        }
        return Y;
    }

    std::vector<Tensor*> parameters() { return {&gamma, &beta}; }
};


struct ReynoldsBatchNorm {
    Tensor gamma, beta;
    int    d_model;
    float  eps      = 1e-5f;
    float  Re_crit  = 1.0f;   // Reynolds criticality threshold (annealed externally)
    float  k_slope  = 5.0f;   // sigmoid steepness for laminar→turbulent transition

    std::vector<float> running_mean;
    std::vector<float> running_var;
    float              ema_decay = 0.99f;
    bool               initialized = false;

    ReynoldsBatchNorm(int d_model_)
        : d_model(d_model_),
          gamma({1, d_model_}, 1.0f),
          beta ({1, d_model_}, 0.0f),
          running_mean(d_model_, 0.0f),
          running_var (d_model_, 1.0f)
    {}

    float reynolds_number(const Tensor& X) const {
        int seq = X.rows(), dim = X.cols();
        float rms_sum = 0.0f, std_sum = 0.0f;

        for (int i = 0; i < seq; ++i) {
            float sq = 0.0f;
            for (int j = 0; j < dim; ++j) sq += X.at(i,j) * X.at(i,j);
            rms_sum += std::sqrt(sq / (dim + 1e-8f));

            float mean = 0.0f, M2 = 0.0f;
            for (int j = 0; j < dim; ++j) {
                float delta = X.at(i,j) - mean;
                mean += delta / (j+1);
                M2   += delta * (X.at(i,j) - mean);
            }
            std_sum += std::sqrt(M2 / dim + eps);
        }

        float rms  = rms_sum / (seq + 1e-8f);
        float std_ = std_sum / (seq + 1e-8f);
        return rms / (std_ + eps);
    }

    float laminar_weight(float Re) const {
        float x = -k_slope * (Re - Re_crit);
        if (x >= 0) {
            float e = std::exp(-x);
            return 1.0f / (1.0f + e);
        } else {
            float e = std::exp(x);
            return e / (1.0f + e);
        }
    }

    Tensor forward(const Tensor& X, bool is_training = true) {
        int seq = X.rows(), dim = X.cols();
        if (seq == 0 || dim == 0) return Tensor(X.shape, 0.0f);
        Tensor Y(X.shape);

        float Re = reynolds_number(X);
        float w_LN = laminar_weight(Re);      // laminar weight
        float w_BN = std::max(0.0f, 1.0f - w_LN);

        Tensor LN_out(X.shape);
        for (int i = 0; i < seq; ++i) {
            float mean = 0.0f, M2 = 0.0f;
            for (int j = 0; j < dim; ++j) {
                float delta = X.at(i,j) - mean;
                mean += delta / (j+1);
                M2   += delta * (X.at(i,j) - mean);
            }
            float inv_std = 1.0f / std::sqrt(M2/dim + eps);
            for (int j = 0; j < dim; ++j)
                LN_out.at(i,j) = (X.at(i,j) - mean) * inv_std;
        }

        std::vector<float> batch_mean(dim, 0.0f), batch_var(dim, 0.0f);
        for (int j = 0; j < dim; ++j) {
            float m = 0.0f, v = 0.0f;
            for (int i = 0; i < seq; ++i) m += X.at(i,j);
            m /= (seq + 1e-8f);
            for (int i = 0; i < seq; ++i) v += (X.at(i,j)-m)*(X.at(i,j)-m);
            v /= (seq + 1e-8f);
            batch_mean[j] = m;
            batch_var[j]  = v;
        }

        if (is_training) {
            for (int j = 0; j < dim; ++j) {
                running_mean[j] = ema_decay * running_mean[j] + (1.0f-ema_decay) * batch_mean[j];
                running_var[j]  = ema_decay * running_var[j]  + (1.0f-ema_decay) * batch_var[j];
            }
            initialized = true;
        }

        const auto& use_mean = (initialized && !is_training) ? running_mean : batch_mean;
        const auto& use_var  = (initialized && !is_training) ? running_var  : batch_var;

        Tensor BN_out(X.shape);
        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < dim; ++j)
                BN_out.at(i,j) = (X.at(i,j) - use_mean[j])
                                / std::sqrt(std::max(0.0f, use_var[j]) + eps);

        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < dim; ++j)
                Y.at(i,j) = gamma[j] * (w_LN * LN_out.at(i,j)
                                       + w_BN * BN_out.at(i,j))
                           + beta[j];

        return Y;
    }

    void anneal_reynolds(int current_step, int total_steps,
                         float Re_start = 2.0f, float Re_end = 0.5f) {
        float ratio = std::min(1.0f, (float)current_step / (float)total_steps);
        float cos_v = 0.5f * (1.0f + std::cos(3.14159265f * ratio));
        Re_crit = Re_end + (Re_start - Re_end) * cos_v;
    }

    std::vector<Tensor*> parameters() { return {&gamma, &beta}; }
};
