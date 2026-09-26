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

// ============================================================
namespace logos_rng {
    // Global generator — default seed 42 (same init every run)
    inline std::mt19937& global_gen() {
        static std::mt19937 gen(42u);
        return gen;
    }

    // Set global seed — call once in main() before model init
    // Example: logos_rng::set_global_seed(42);
    inline void set_global_seed(unsigned seed) {
        global_gen().seed(seed);
    }

    // Draw N values from [lo, hi] using global generator
    inline void fill_uniform(std::vector<float>& buf, float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        for (float& v : buf) v = dist(global_gen());
    }

    // Draw N values using a LOCAL generator (isolated, reproducible)
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

    // ── Constructors ──────────────────────────────────────────
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

    // ── Shape utilities ───────────────────────────────────────
    int rows() const { return shape.size() >= 2 ? shape[shape.size()-2] : 1; }
    int cols() const { return shape.empty() ? 0 : shape.back(); }
    int ndim() const { return static_cast<int>(shape.size()); }

    // ── BUG 15 FIX: bounds check always active ───────────────
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

    // ── 1D access (unchecked — hot paths) ────────────────────
    float& operator[](int i)       { return data[i]; }
    float  operator[](int i) const { return data[i]; }

    // ── Fill ─────────────────────────────────────────────────
    void fill(float val) { std::fill(data.begin(), data.end(), val); }
    void zero()          { fill(0.0f); }

    // BUG 8 FIX: Deterministic random fill
    // local_seed < 0 (default) → use global seeded RNG (recommended)
    // local_seed >= 0          → use isolated local RNG for this tensor
    //
    // For reproducible training:
    //   Call logos_rng::set_global_seed(42) once in main() before model init.
    //   All fill_random() calls with no local_seed will then be deterministic.
    //
    // Pehle: rand() → non-deterministic, thread-unsafe, srand() ignored in training
    // Ab:    logos_rng::global_gen() → seeded, deterministic, consistent
    void fill_random(float lo = -0.1f, float hi = 0.1f, int local_seed = -1) {
        if (local_seed >= 0) {
            logos_rng::fill_uniform_local(data, lo, hi,
                                          static_cast<unsigned>(local_seed));
        } else {
            logos_rng::fill_uniform(data, lo, hi);
        }
    }

    // ── Element-wise ops ──────────────────────────────────────
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

    // ── BUG 2 FIX: reshape — avoids unnecessary copy ─────────
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

    // ── Transpose (2D only) ───────────────────────────────────
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

    // ── Debug print ───────────────────────────────────────────
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

    // ── Statistics ────────────────────────────────────────────
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

// ── Riemannian Metric for Parameter Space ─────────────────────
// Physics: standard Euclidean distance in weight space treats all
//   parameter directions as equally important. This is geometrically
//   naive — the loss landscape is curved (non-Euclidean manifold).
//
// Riemannian geometry:
//   The parameter space θ ∈ ℝⁿ is equipped with a metric tensor G(θ)
//   that defines local distances as:
//     ds² = dθᵀ G(θ) dθ   (infinitesimal arc length on the manifold)
//
//   In standard gradient descent: G = I (flat Euclidean)
//   In natural gradient:          G = Fisher information matrix F(θ)
//   In Riemannian gradient:       G = any positive-definite curvature estimate
//
// This implementation: diagonal Riemannian metric from Hessian diagonal
//   G_ii ≈ |∂²L/∂θᵢ²|   (absolute curvature per parameter)
//   Estimated via finite differences: |g(θ+ε) - g(θ)| / ε
//   Simple, cheap, and gives per-parameter curvature information.
//
// Uses:
//   (1) riemannian_distance()  — geodesic distance between two param vectors
//   (2) riemannian_gradient()  — precondition gradient by G⁻¹ (natural step)
//   (3) parallel_transport()   — move a gradient vector along a path on the manifold
//       (first-order approx: g_transported ≈ g - (g·Δθ/||Δθ||²) * Δθ)
//
// Note: For full Hessian-based metric, see compute_metric_from_grads() below.
//       For Fisher-based metric, use NaturalGradientOptimizer in PhysicsOpt.hpp.
struct RiemannianMetric {
    std::vector<float> metric_diag;  // G_ii — diagonal metric tensor
    float              damping;       // λ > 0 ensures G + λI is PD (Tikhonov)
    int                dim;

    explicit RiemannianMetric(int dimension, float damp = 1e-4f)
        : metric_diag(dimension, 1.0f),   // init to identity (flat Euclidean)
          damping(damp), dim(dimension)
    {}

    // Update metric diagonal from a sequence of gradient vectors
    // G_ii ← β * G_ii + (1-β) * g_i²   (EMA of squared gradients)
    // This is the empirical Fisher / AdaGrad-style curvature estimate.
    // Call this after each backward pass with the current gradient.
    void update(const Tensor& grad, float beta = 0.95f) {
        if ((int)grad.total_size > dim) return;  // safety check
        for (int i = 0; i < grad.total_size && i < dim; ++i) {
            float g2 = grad.data[i] * grad.data[i];
            metric_diag[i] = beta * metric_diag[i] + (1.0f - beta) * g2;
        }
    }

    // Batch update from multiple parameter tensors (all params at once)
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

    // Riemannian (geodesic) distance between two parameter points θ₁ and θ₂
    // d²(θ₁,θ₂) = (θ₁-θ₂)ᵀ G (θ₁-θ₂) = Σᵢ G_ii * (θ₁ᵢ - θ₂ᵢ)²
    // Returns the geodesic length (scalar), not squared distance.
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

    // Tensor overload for convenience
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

    // Precondition a gradient by the inverse metric: g̃ = G⁻¹ g
    // g̃_i = g_i / (G_ii + λ)   [diagonal inverse = element-wise division]
    // This transforms the standard gradient into the natural gradient direction.
    // The result is the steepest ascent direction in Riemannian metric distance.
    Tensor riemannian_gradient(const Tensor& grad) const {
        return riemannian_gradient_at(grad, 0);
    }

    // [WIRED — offset variant] update_from_params() builds the metric by
    // concatenating ALL parameter tensors into one flat index space (see
    // its `offset` accumulation above). To precondition a single tensor's
    // gradient consistently with that global metric, the caller must supply
    // the same offset it used when building metric_diag for this tensor.
    // Without this overload, every tensor would incorrectly read
    // metric_diag[0..size) regardless of where its params actually live in
    // the flattened space — silently wrong once more than one tensor exists.
    Tensor riemannian_gradient_at(const Tensor& grad, int offset) const {
        Tensor g_nat(grad.shape);
        int n = std::min(grad.total_size, std::max(0, dim - offset));
        for (int i = 0; i < n; ++i) {
            float G_ii = metric_diag[offset + i] + damping;
            g_nat.data[i] = grad.data[i] / G_ii;
        }
        // Any elements beyond metric coverage pass through unscaled (safety).
        for (int i = n; i < grad.total_size; ++i) g_nat.data[i] = grad.data[i];
        return g_nat;
    }

    // Parallel transport (first-order approximation):
    // When moving from θ to θ + Δθ, a tangent vector v (gradient) must be
    // transported along the geodesic to remain "parallel" on the manifold.
    //
    // Exact parallel transport requires solving ODEs; first-order approx:
    //   v_transported ≈ v - (v·Δθ / (||Δθ||²_G + ε)) * Δθ
    //   where ||Δθ||²_G = ΔθᵀGΔθ  (Riemannian norm of step)
    //
    // This removes the component of v in the direction of motion,
    // keeping v on the tangent space of the new point on the manifold.
    Tensor parallel_transport(const Tensor& v, const Tensor& delta_theta) const {
        // Compute Riemannian inner product <v, Δθ>_G = Σᵢ G_ii * v_i * Δθ_i
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

    // Riemannian norm of a vector: ||v||_G = sqrt(vᵀ G v)
    float riemannian_norm(const Tensor& v) const {
        float n2 = 0.0f;
        int   n  = std::min(v.total_size, dim);
        for (int i = 0; i < n; ++i) {
            float G_ii = metric_diag[i] + damping;
            n2 += G_ii * v.data[i] * v.data[i];
        }
        return std::sqrt(n2);
    }

    // Log current metric statistics (for diagnostics)
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
