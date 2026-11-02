#pragma once
#include <vector>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <random>
#include <atomic>

namespace logos_rng {
    inline std::mt19937& global_gen() {
        static std::mt19937 gen(42u);
        return gen;
    }

    inline void set_global_seed(unsigned seed) {
        global_gen().seed(seed);
    }

    inline void fill_uniform(std::vector<float>& buf, float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        for (float& v : buf) v = dist(global_gen());
    }

    inline void fill_uniform_local(std::vector<float>& buf, float lo, float hi,
                                   unsigned seed) {
        std::mt19937 local_gen(seed);
        std::uniform_real_distribution<float> dist(lo, hi);
        for (float& v : buf) v = dist(local_gen);
    }
}

class Tensor {
public:
    std::vector<float> data;
    std::vector<int>   shape;
    int total_size = 0;

    Tensor() = default;

    explicit Tensor(std::vector<int> shape_, float fill = 0.0f)
        : shape(std::move(shape_))
    {
        total_size = 1;
        for (int d : shape) {
            if (d <= 0) throw std::invalid_argument("Shape dimension must be > 0");
            total_size *= d;
        }
        data.reserve(total_size);
        data.assign(total_size, fill);
    }

    int rows() const { return shape.size() >= 2 ? shape[shape.size()-2] : 1; }
    int cols() const { return shape.empty() ? 0 : shape.back(); }
    int ndim() const { return static_cast<int>(shape.size()); }

    float& at(int r, int c) {
        if (r < 0 || c < 0 || r >= rows() || c >= cols())
            throw std::out_of_range(
                "Tensor::at(" + std::to_string(r) + "," + std::to_string(c) +
                ") out of range for (" + std::to_string(rows()) + "," +
                std::to_string(cols()) + ")");
        return data[r * cols() + c];
    }
    float at(int r, int c) const {
        if (r < 0 || c < 0 || r >= rows() || c >= cols())
            throw std::out_of_range(
                "Tensor::at(" + std::to_string(r) + "," + std::to_string(c) +
                ") out of range for (" + std::to_string(rows()) + "," +
                std::to_string(cols()) + ")");
        return data[r * cols() + c];
    }

    float& operator[](int i)       { return data[i]; }
    float  operator[](int i) const { return data[i]; }

    void fill(float val) { std::fill(data.begin(), data.end(), val); }
    void zero()          { fill(0.0f); }

    void fill_random(float lo = -0.1f, float hi = 0.1f, int local_seed = -1) {
        if (local_seed >= 0) {
            logos_rng::fill_uniform_local(data, lo, hi,
                                          static_cast<unsigned>(local_seed));
        } else {
            logos_rng::fill_uniform(data, lo, hi);
        }
    }

    Tensor operator+(const Tensor& other) const {
        if (total_size != other.total_size)
            throw std::invalid_argument("Tensor::operator+ size mismatch");
        Tensor result(shape);
        for (int i = 0; i < total_size; ++i)
            result.data[i] = data[i] + other.data[i];
        return result;
    }

    Tensor operator*(float scalar) const {
        Tensor result(shape);
        for (int i = 0; i < total_size; ++i)
            result.data[i] = data[i] * scalar;
        return result;
    }

    Tensor& operator+=(const Tensor& other) {
        if (total_size != other.total_size)
            throw std::invalid_argument("Tensor::operator+= size mismatch");
        for (int i = 0; i < total_size; ++i)
            data[i] += other.data[i];
        return *this;
    }

    Tensor reshape(std::vector<int> new_shape) const {
        int new_total = 1;
        for (int d : new_shape) new_total *= d;
        if (new_total != total_size)
            throw std::invalid_argument(
                "reshape: elements mismatch (" + std::to_string(total_size) +
                " → " + std::to_string(new_total) + ")");
        Tensor result;
        result.data       = data;
        result.shape      = std::move(new_shape);
        result.total_size = total_size;
        return result;
    }

    Tensor& reshape_inplace(std::vector<int> new_shape) {
        int new_total = 1;
        for (int d : new_shape) new_total *= d;
        if (new_total != total_size)
            throw std::invalid_argument("reshape_inplace: elements mismatch");
        shape = std::move(new_shape);
        return *this;
    }

    Tensor transpose() const {
        if (ndim() != 2)
            throw std::invalid_argument("transpose: only 2D tensors supported");
        int R = rows(), C = cols();
        Tensor result({C, R});
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                result.at(c, r) = at(r, c);
        return result;
    }

    void print(const std::string& name = "Tensor", int max_rows = 4, int max_cols = 8) const {
        std::cout << "[" << name << "] shape=(";
        for (int i = 0; i < (int)shape.size(); ++i)
            std::cout << shape[i] << (i+1<(int)shape.size() ? "," : "");
        std::cout << ")\n";
        int R = std::min(rows(), max_rows);
        int C = std::min(cols(), max_cols);
        for (int r = 0; r < R; ++r) {
            std::cout << "  [ ";
            for (int c = 0; c < C; ++c) std::cout << at(r, c) << " ";
            if (cols() > max_cols) std::cout << "...";
            std::cout << "]\n";
        }
        if (rows() > max_rows) std::cout << "  ...\n";
    }

    float mean() const {
        if (total_size == 0) return 0.0f;
        return std::accumulate(data.begin(), data.end(), 0.0f) / total_size;
    }
    float max_val() const {
        if (data.empty()) throw std::runtime_error("max_val on empty tensor");
        return *std::max_element(data.begin(), data.end());
    }
    float min_val() const {
        if (data.empty()) throw std::runtime_error("min_val on empty tensor");
        return *std::min_element(data.begin(), data.end());
    }
    bool has_nan() const {
        for (float v : data) if (std::isnan(v) || std::isinf(v)) return true;
        return false;
    }
};

struct RiemannianMetric {
    std::vector<float> metric_diag;  // G_ii — diagonal metric tensor
    float              damping;       // λ > 0 ensures G + λI is PD (Tikhonov)
    int                dim;

    explicit RiemannianMetric(int dimension, float damp = 1e-4f)
        : metric_diag(dimension, 1.0f),   // init to identity (flat Euclidean)
          damping(damp), dim(dimension)
    {}

    void update(const Tensor& grad, float beta = 0.95f) {
        if ((int)grad.total_size > dim) return;  // safety check
        for (int i = 0; i < grad.total_size && i < dim; ++i) {
            float g2 = grad.data[i] * grad.data[i];
            metric_diag[i] = beta * metric_diag[i] + (1.0f - beta) * g2;
        }
    }

    void update_from_params(const std::vector<Tensor*>& grads, float beta = 0.95f) {
        int offset = 0;
        for (const auto* g : grads) {
            for (int i = 0; i < g->total_size && offset+i < dim; ++i) {
                float g2 = g->data[i] * g->data[i];
                metric_diag[offset+i] = beta * metric_diag[offset+i]
                                       + (1.0f - beta) * g2;
            }
            offset += g->total_size;
        }
    }

    float riemannian_distance(const std::vector<float>& theta1,
                               const std::vector<float>& theta2) const {
        float d2 = 0.0f;
        int   n  = std::min({(int)theta1.size(), (int)theta2.size(), dim});
        for (int i = 0; i < n; ++i) {
            float diff = theta1[i] - theta2[i];
            float G_ii = metric_diag[i] + damping;  // G + λI ensures PD
            d2 += G_ii * diff * diff;
        }
        return std::sqrt(d2);
    }

    float riemannian_distance(const Tensor& theta1, const Tensor& theta2) const {
        float d2 = 0.0f;
        int   n  = std::min({theta1.total_size, theta2.total_size, dim});
        for (int i = 0; i < n; ++i) {
            float diff = theta1.data[i] - theta2.data[i];
            float G_ii = metric_diag[i] + damping;
            d2 += G_ii * diff * diff;
        }
        return std::sqrt(d2);
    }

    Tensor riemannian_gradient(const Tensor& grad) const {
        return riemannian_gradient_at(grad, 0);
    }

    Tensor riemannian_gradient_at(const Tensor& grad, int offset) const {
        Tensor g_nat(grad.shape);
        int n = std::min(grad.total_size, std::max(0, dim - offset));
        for (int i = 0; i < n; ++i) {
            float G_ii = metric_diag[offset + i] + damping;
            g_nat.data[i] = grad.data[i] / G_ii;
        }
        for (int i = n; i < grad.total_size; ++i) g_nat.data[i] = grad.data[i];
        return g_nat;
    }

    Tensor parallel_transport(const Tensor& v, const Tensor& delta_theta) const {
        float vdot_G = 0.0f;
        float ddot_G = 0.0f;   // ||Δθ||²_G
        int   n      = std::min({v.total_size, delta_theta.total_size, dim});
        for (int i = 0; i < n; ++i) {
            float G_ii = metric_diag[i] + damping;
            vdot_G += G_ii * v.data[i] * delta_theta.data[i];
            ddot_G += G_ii * delta_theta.data[i] * delta_theta.data[i];
        }

        float scale = vdot_G / (ddot_G + 1e-10f);

        Tensor transported(v.shape);
        for (int i = 0; i < n; ++i)
            transported.data[i] = v.data[i] - scale * delta_theta.data[i];
        return transported;
    }

    float riemannian_norm(const Tensor& v) const {
        float n2 = 0.0f;
        int   n  = std::min(v.total_size, dim);
        for (int i = 0; i < n; ++i) {
            float G_ii = metric_diag[i] + damping;
            n2 += G_ii * v.data[i] * v.data[i];
        }
        return std::sqrt(n2);
    }

    void log_state() const {
        float min_g = metric_diag[0], max_g = metric_diag[0], sum = 0.0f;
        for (float g : metric_diag) {
            min_g = std::min(min_g, g);
            max_g = std::max(max_g, g);
            sum  += g;
        }
        std::cout << "RiemannianMetric | dim=" << dim
                  << " | G_min=" << min_g
                  << " | G_max=" << max_g
                  << " | G_mean=" << (sum / dim)
                  << " | λ=" << damping << "\n";
    }
};
